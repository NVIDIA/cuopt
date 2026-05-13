#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Aggregate per-matrix PR test summaries into a Markdown body for the PR
classification comment.

Each PR test job runs ``nightly_report.py --mode pr`` which writes a
per-matrix summary JSON to::

    s3://bucket/ci_test_reports/pr/run-${GITHUB_RUN_ID}/{test_type}-{matrix}.json

This script downloads them, merges with the shared aggregator helpers, and
emits two Markdown sections:

  - **NEW failures** — failures introduced by this PR (not in nightly
    history, or only present as resolved-and-not-flaky).
  - **KNOWN issues** — pre-existing breakage (active on nightly) or known
    flakes (flagged on nightly, or flaked in this PR run).

The output Markdown is prefixed with a hidden marker comment so the
comment poster (``ci/pr_summary.sh``) can find and update an existing
comment in place.

If nothing failed or flaked across the run, this script writes an empty
file and the poster skips commenting.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_common import (  # noqa: E402
    aggregate_summaries,
    download_summaries,
    load_local_summaries,
)
from pr_comment_helper import COMMENT_MARKER  # noqa: E402

# Maximum total comment body size we are willing to post.  GitHub allows
# ~65k characters per comment, but we cap earlier and truncate the failure
# tables so the comment stays readable.
MAX_BODY_CHARS = 60000
MAX_ROWS_PER_BUCKET = 80
MAX_ERROR_SNIPPET = 600
# Crash entries get their full message in a code block, capped only at a
# generous limit since the diagnostic line is the whole point of the entry.
MAX_CRASH_MESSAGE_CHARS = 2000

# Crashes write a JUnit case named "PROCESS_CRASH" with a message
# containing "crashed with SIG..." (see ci/utils/crash_helpers.sh from
# PR #1191).  Match either fingerprint defensively.
_CRASH_NAME = "PROCESS_CRASH"
_CRASH_MESSAGE_RE = re.compile(r"crashed with SIG[A-Z]+", re.IGNORECASE)


def _is_crash(entry):
    if entry.get("name") == _CRASH_NAME:
        return True
    return bool(_CRASH_MESSAGE_RE.search(entry.get("message", "") or ""))


def _split_crashes(failures):
    """Partition a failures list into ``(crashes, non_crash)``."""
    crashes = []
    non_crash = []
    for entry in failures:
        (crashes if _is_crash(entry) else non_crash).append(entry)
    return crashes, non_crash


def _short_msg(msg, limit=300):
    """Single-line summary of an error message for table cells."""
    if not msg:
        return ""
    lines = [ln for ln in msg.splitlines() if ln.strip()]
    summary = lines[-1] if lines else ""
    if len(summary) > limit:
        summary = summary[: limit - 1] + "…"
    return summary.replace("|", "\\|")


def _details_block(msg):
    """Render an error message as a collapsible <details> block."""
    if not msg:
        return ""
    snippet = msg.strip()
    if len(snippet) > MAX_ERROR_SNIPPET:
        snippet = snippet[:MAX_ERROR_SNIPPET] + "\n…[truncated]"
    return (
        f"<details><summary>error</summary>\n\n```\n{snippet}\n```\n</details>"
    )


def _classify_known_subgroups(recurring, flaky):
    """Split the KNOWN bucket into the three sub-groups for the comment.

    Returns ``(broken_on_nightly, known_flaky_nightly, flaked_in_pr_run)``.
    Each entry retains its full per-matrix context.
    """
    broken_on_nightly = []
    known_flaky_nightly = []
    flaked_in_pr_run = []

    for entry in recurring:
        cls = entry.get("pr_classification", "")
        if cls == "known_recurring":
            broken_on_nightly.append(entry)
        elif cls == "known_flaky_nightly":
            known_flaky_nightly.append(entry)
        else:
            broken_on_nightly.append(entry)

    for entry in flaky:
        cls = entry.get("pr_classification", "")
        if cls == "known_flaky_nightly":
            known_flaky_nightly.append(entry)
        elif cls == "known_recurring":
            broken_on_nightly.append(entry)
        else:
            flaked_in_pr_run.append(entry)

    return broken_on_nightly, known_flaky_nightly, flaked_in_pr_run


def _matrix_grid_table(grid):
    if not grid:
        return ""
    lines = [
        "| Test type | Matrix | Status | Passed | Failed | Flaky | Skipped |",
        "|-----------|--------|--------|--------|--------|-------|---------|",
    ]
    badge_for = {
        "passed": "PASS",
        "failed-new": "NEW FAIL",
        "failed-recurring": "RECURRING",
        "flaky": "FLAKY",
        "no-results": "NO DATA",
    }
    for g in grid:
        c = g.get("counts", {})
        lines.append(
            f"| {g['test_type']} | `{g['matrix_label']}` | "
            f"{badge_for.get(g['status'], g['status'])} | "
            f"{c.get('passed', 0)} | {c.get('failed', 0)} | "
            f"{c.get('flaky', 0)} | {c.get('skipped', 0)} |"
        )
    return "\n".join(lines)


def _failure_table(entries, columns, row_fn, cap=MAX_ROWS_PER_BUCKET):
    if not entries:
        return ""
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for entry in entries[:cap]:
        lines.append(row_fn(entry))
    if len(entries) > cap:
        lines.append(f"\n_…and {len(entries) - cap} more not shown._")
    return "\n".join(lines)


def build_comment_body(
    agg, target_branch, github_run_url, sha="", run_date=""
):
    """Build the Markdown body for the sticky PR comment.

    Returns an empty string if there is nothing worth commenting on.
    """
    new_failures = agg["all_new_failures"]
    recurring = agg["all_recurring_failures"]
    flaky = agg["all_flaky_tests"]

    if not new_failures and not recurring and not flaky:
        return ""

    # Pull crashes out so they render in a dedicated CAUTION block above
    # the normal NEW/KNOWN tables and don't get drowned out by ordinary
    # assertion failures.
    new_crashes, new_failures = _split_crashes(new_failures)
    recurring_crashes, recurring = _split_crashes(recurring)
    all_crashes = new_crashes + recurring_crashes

    broken_on_nightly, known_flaky_nightly, flaked_in_pr_run = (
        _classify_known_subgroups(recurring, flaky)
    )

    parts = [COMMENT_MARKER]
    parts.append("## PR Test Classification")
    parts.append("")

    headline = []
    if all_crashes:
        headline.append(f"**{len(all_crashes)} CRASH(es)**")
    if new_failures:
        headline.append(f"**{len(new_failures)} NEW** failure(s)")
    known_total = (
        len(broken_on_nightly)
        + len(known_flaky_nightly)
        + len(flaked_in_pr_run)
    )
    if known_total:
        headline.append(f"**{known_total} KNOWN** issue(s)")
    if headline:
        parts.append(" • ".join(headline))
        parts.append("")

    meta = []
    if target_branch:
        meta.append(f"Compared against nightly history for `{target_branch}`")
    if sha:
        meta.append(f"PR head: `{sha[:12]}`")
    if run_date:
        meta.append(f"Run date: {run_date}")
    if github_run_url:
        meta.append(f"[Workflow run]({github_run_url})")
    if meta:
        parts.append(" · ".join(meta))
        parts.append("")

    grid_md = _matrix_grid_table(agg["matrix_grid"])
    if grid_md:
        parts.append("<details><summary>Per-matrix status</summary>\n")
        parts.append(grid_md)
        parts.append("\n</details>")
        parts.append("")

    # --- CRASHES (top of comment, GitHub red-alert callout) ---
    if all_crashes:
        parts.append("> [!CAUTION]")
        parts.append(
            "> **CRASHES detected — a test process was terminated by a signal mid-run.**"
        )
        parts.append(
            "> These need urgent investigation.  The JUnit XML was not "
            "finalized, so the specific test that triggered the crash "
            "may not be identified; check the workflow run log for the "
            "last test invoked before the signal."
        )
        parts.append("")
        # Collapse per-crash details under a hidden tab — the CAUTION
        # block is the headline; details are one click away.
        crash_word = "crash" if len(all_crashes) == 1 else "crashes"
        parts.append("<details>")
        parts.append(
            f"<summary><strong>{len(all_crashes)} {crash_word}"
            " — click to expand details</strong></summary>"
        )
        parts.append("")  # blank line so the body renders as Markdown
        for entry in all_crashes:
            heading_tag = (
                "NEW" if entry.get("pr_classification") == "new" else "KNOWN"
            )
            parts.append(
                f"#### `{entry.get('suite', '?')}` — "
                f"`{entry.get('name', 'PROCESS_CRASH')}` "
                f"_[{entry['test_type']} / {entry['matrix_label']}]_ "
                f"— {heading_tag}"
            )
            msg = (entry.get("message") or "").strip()
            if msg:
                if len(msg) > MAX_CRASH_MESSAGE_CHARS:
                    msg = msg[:MAX_CRASH_MESSAGE_CHARS] + "\n…[truncated]"
                parts.append("")
                parts.append("```")
                parts.append(msg)
                parts.append("```")
            parts.append("")
        parts.append("</details>")
        parts.append("")

    # --- NEW failures (red CAUTION callout as the section header) ---
    if new_failures:
        parts.append("> [!CAUTION]")
        parts.append(
            f"> **NEW failures ({len(new_failures)}) — likely introduced by this PR**"
        )
        parts.append("")
        parts.append(
            _failure_table(
                new_failures,
                ["Test type", "Matrix", "Suite", "Test", "Error"],
                lambda e: (
                    f"| {e['test_type']} | `{e['matrix_label']}` | "
                    f"{e['suite']} | `{e['name']}` | "
                    f"{_short_msg(e.get('message', ''))} |"
                ),
            )
        )
        parts.append("")

    # --- KNOWN issues ---
    if known_total:
        parts.append("### KNOWN issues (pre-existing, not caused by this PR)")
        parts.append("")

        if broken_on_nightly:
            parts.append("**Already broken on nightly** (recurring)")
            parts.append("")
            parts.append(
                _failure_table(
                    broken_on_nightly,
                    [
                        "Test type",
                        "Matrix",
                        "Suite",
                        "Test",
                        "First seen",
                        "Failure count",
                        "Error",
                    ],
                    lambda e: (
                        f"| {e['test_type']} | `{e['matrix_label']}` | "
                        f"{e['suite']} | `{e['name']}` | "
                        f"{e.get('first_seen', 'unknown')} | "
                        f"{e.get('failure_count', '?')} | "
                        f"{_short_msg(e.get('message', ''))} |"
                    ),
                )
            )
            parts.append("")

        if known_flaky_nightly:
            parts.append("**Known flaky on nightly**")
            parts.append("")
            parts.append(
                _failure_table(
                    known_flaky_nightly,
                    [
                        "Test type",
                        "Matrix",
                        "Suite",
                        "Test",
                        "First seen",
                        "Error",
                    ],
                    lambda e: (
                        f"| {e['test_type']} | `{e['matrix_label']}` | "
                        f"{e['suite']} | `{e['name']}` | "
                        f"{e.get('first_seen', 'unknown')} | "
                        f"{_short_msg(e.get('message', ''))} |"
                    ),
                )
            )
            parts.append("")

        if flaked_in_pr_run:
            parts.append(
                "**Flaked in this PR run** (passed on retry; not previously known to flake)"
            )
            parts.append("")
            parts.append(
                _failure_table(
                    flaked_in_pr_run,
                    [
                        "Test type",
                        "Matrix",
                        "Suite",
                        "Test",
                        "Retries",
                        "Error",
                    ],
                    lambda e: (
                        f"| {e['test_type']} | `{e['matrix_label']}` | "
                        f"{e['suite']} | `{e['name']}` | "
                        f"{e.get('retry_count', '?')} | "
                        f"{_short_msg(e.get('message', ''))} |"
                    ),
                )
            )
            parts.append("")

    parts.append(
        "_Classification compares each failure against the most recent "
        "nightly history for the target branch.  Tests passed on retry "
        "via `pytest-rerunfailures` are reported as flaky._"
    )

    body = "\n".join(parts)
    if len(body) > MAX_BODY_CHARS:
        body = body[: MAX_BODY_CHARS - 200] + (
            "\n\n…_comment truncated; see workflow run for full details._"
        )
    return body


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-matrix PR test summaries into a Markdown PR comment."
    )
    parser.add_argument(
        "--s3-pr-summaries-prefix",
        default="",
        help=(
            "S3 prefix where ``nightly_report.py --mode pr`` uploaded "
            "per-matrix summaries for this run.  Example: "
            "s3://bucket/ci_test_reports/pr/run-12345/"
        ),
    )
    parser.add_argument(
        "--local-summaries-dir",
        default="",
        help="Local directory of summaries (for testing without S3).",
    )
    parser.add_argument(
        "--output-dir",
        default="aggregate-output",
        help="Directory to write pr_comment.md and consolidated.json into.",
    )
    parser.add_argument(
        "--target-branch",
        default=os.environ.get("GITHUB_BASE_REF", "main"),
        help="PR target branch — surfaced in the comment for context.",
    )
    parser.add_argument(
        "--sha",
        default=os.environ.get("GITHUB_SHA", ""),
        help="PR head SHA — surfaced in the comment for context.",
    )
    parser.add_argument(
        "--github-run-url",
        default="",
        help="Workflow run URL — linked from the comment footer.",
    )
    parser.add_argument(
        "--run-date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date the run started (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.local_summaries_dir:
        summaries = load_local_summaries(args.local_summaries_dir)
    elif args.s3_pr_summaries_prefix:
        summaries = download_summaries(
            args.s3_pr_summaries_prefix, output_dir / "summaries"
        )
    else:
        print(
            "ERROR: provide --s3-pr-summaries-prefix or --local-summaries-dir.",
            file=sys.stderr,
        )
        return 2

    if not summaries:
        print("No PR per-matrix summaries found; nothing to comment on.")
        (output_dir / "pr_comment.md").write_text("")
        return 0

    agg = aggregate_summaries(summaries)
    body = build_comment_body(
        agg,
        target_branch=args.target_branch,
        github_run_url=args.github_run_url,
        sha=args.sha,
        run_date=args.run_date,
    )

    (output_dir / "pr_comment.md").write_text(body)
    consolidated = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_branch": args.target_branch,
        "sha": args.sha,
        "run_date": args.run_date,
        "totals": agg["totals"],
        "matrix_grid": agg["matrix_grid"],
        "new_failures": agg["all_new_failures"],
        "recurring_failures": agg["all_recurring_failures"],
        "flaky_tests": agg["all_flaky_tests"],
    }
    (output_dir / "pr_consolidated.json").write_text(
        json.dumps(consolidated, indent=2) + "\n"
    )

    if not body:
        print("All tests passed (no failures or flakes); skipping PR comment.")
    else:
        print(
            f"PR comment body written to {output_dir / 'pr_comment.md'} "
            f"({len(body)} chars)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
