# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Slack Block Kit payloads from a consolidated nightly summary JSON.

Prints one JSON payload per line to stdout:
  - Line 1: main channel message (thread parent)
  - Lines 2+: thread replies (per-workflow details, failed job links)

Usage:
    python3 generate_slack_payloads.py <summary.json> [presigned_report_url] [presigned_dashboard_url]
"""

import json
import os
import re
import sys

# Same window the per-job report uses for "failed N of the last M nights".
FAIL_WINDOW_DAYS = int(os.environ.get("CUOPT_FAIL_WINDOW_DAYS", 14))

# Failures in one workflow that share a reason collapse into one line once
# there are at least this many of them.
GROUP_MIN = 3

# Maximum lines listed per category and workflow before "+N more".
MAX_LINES = 10

_BUILD_PREFIX = re.compile(r"^.*?/work/")
_LOCATION = re.compile(r"^(\S+\.(?:cu|cuh|cpp|hpp|h|py|pyx)):(\d+)\b")


def _esc(text):
    """Escape Slack mrkdwn special characters in dynamic text."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _job_prefix(job):
    """Extract workflow prefix from a GitHub Actions job name."""
    name = job.get("name", "unknown")
    return name.split(" / ")[0] if " / " in name else name


def clean_message(message, limit=140):
    """Turn a raw failure message into a short reason plus ``file:line``.

    gtest failure text starts with the absolute build path of the source file
    and the line number, for example
    ``/tmp/conda-bld-output/bld/rattler-build_x/work/cpp/tests/a_test.cu:42``
    followed by the assertion. Truncating that to 100 characters left only the
    path. This drops the build prefix, keeps ``a_test.cu:42`` and the first
    line that says what went wrong. For pytest, the first ``E`` line is used.
    """
    lines = [
        _BUILD_PREFIX.sub("", ln.strip())
        for ln in str(message).splitlines()
        if ln.strip()
    ]
    if not lines:
        return ""

    location = ""
    tail = ""  # text after "file:line:" on the same line, used as a fallback
    rest = []
    for ln in lines:
        match = _LOCATION.match(ln)
        if match and not location:
            location = f"{os.path.basename(match.group(1))}:{match.group(2)}"
            tail = ln[match.end() :].lstrip(": ").strip()
        else:
            rest.append(ln)

    reason = ""
    for ln in rest:  # pytest: the first "E" line is the assertion
        if ln.startswith("E "):
            reason = ln[2:].strip()
            break
    if not reason and rest:
        reason = rest[0]
        # gtest: "Expected equality of these values:" is followed by the values
        if reason.endswith(":") and len(rest) > 1:
            reason = " ".join([reason] + [r.strip() for r in rest[1:3]])

    reason = reason or tail
    if not location:
        return reason[:limit]
    # Truncate the reason, never the location.
    suffix = f" @ {location}"
    if not reason:
        return location[:limit]
    return reason[: max(0, limit - len(suffix))] + suffix


def reason_signature(message):
    """Normalised reason used to group failures that share a cause.

    Drops the source location and digits so the same assertion at different
    sizes or indices groups together. An empty reason never groups.
    """
    reason = clean_message(message).split(" @ ")[0]
    return re.sub(r"\d+(\.\d+)?", "#", reason).strip().lower()


def _dedupe(entries):
    """Drop repeated (name, matrix) entries, keeping the first."""
    seen = set()
    out = []
    for e in entries:
        key = (
            e.get("suite"),
            e.get("classname"),
            e.get("name"),
            e.get("matrix_label"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _nights(entry):
    """' · N/14 nights' once a test has failed on more than one night."""
    n = entry.get("nights_failed", 1)
    return f" \u00b7 {n}/{FAIL_WINDOW_DAYS} nights" if n and n >= 2 else ""


def _matrix_summary(entries):
    labels = sorted({e.get("matrix_label", "") for e in entries} - {""})
    if len(labels) <= 2:
        return ", ".join(labels)
    return ", ".join(labels[:2]) + f", +{len(labels) - 2} more"


def issue_lines(entries, tag, detail=None, group_detail=None, limit=MAX_LINES):
    """Slack lines for failing or flaky tests of one workflow.

    Entries that share a reason collapse into one line once there are
    GROUP_MIN of them. ``tag`` is an emoji string or a function of the entry.
    ``detail`` formats the suffix of a single line and ``group_detail`` the
    suffix of a collapsed line. Returns (lines, hidden) where hidden is the
    number of lines that did not fit in ``limit``.
    """
    unique = _dedupe(entries)
    groups = {}
    for e in unique:
        sig = reason_signature(e.get("message", ""))
        groups.setdefault(sig or f"\0{e.get('name')}", []).append(e)

    lines = []
    for members in groups.values():
        if len(members) >= GROUP_MIN:
            reason = _esc(clean_message(members[0].get("message", "")))
            names = ", ".join(
                f"`{_esc(m.get('name', 'unknown'))}`" for m in members[:3]
            )
            more = f" +{len(members) - 3} more" if len(members) > 3 else ""
            matrix = _esc(_matrix_summary(members))
            suffix = group_detail(members) if group_detail else ""
            # Use a new member for the marker so a group with a new flaky
            # test is not shown as already known.
            marker = next((m for m in members if m.get("is_new")), members[0])
            group_tag = tag(marker) if callable(tag) else tag
            lines.append(
                f"{group_tag}  *{len(members)} tests* ({matrix}) "
                f"\u2014 {reason}{suffix}"
                f"\n      e.g. {names}{more}"
            )
        else:
            for m in members:
                line_tag = tag(m) if callable(tag) else tag
                lines.append(
                    f"{line_tag}  `{_esc(m.get('name', 'unknown'))}` "
                    f"({_esc(m.get('matrix_label', ''))})"
                    + (detail(m) if detail else "")
                )
    return lines[:limit], max(0, len(lines) - limit)


def repo_base(github_run_url):
    """``https://github.com/org/repo`` from a run URL, or empty."""
    return github_run_url.split("/actions/runs/")[0] if github_run_url else ""


def make_payload(blocks):
    return json.dumps(
        {
            "username": "cuOpt Nightly Bot",
            "icon_emoji": ":robot_face:",
            "blocks": blocks,
        }
    )


def main():
    summary_path = sys.argv[1]
    presigned_report_url = sys.argv[2] if len(sys.argv) > 2 else ""
    presigned_dashboard_url = sys.argv[3] if len(sys.argv) > 3 else ""

    with open(summary_path) as f:
        d = json.load(f)

    branch = d.get("branch", "main")
    date = d.get("date", "unknown")
    github_run_url = d.get("github_run_url", "")
    jobs = d.get("job_summary", {})
    totals = d.get("test_totals", {})
    grid = d.get("matrix_grid", [])
    has_new = d.get("has_new_failures", False)
    has_new_flaky = d.get("has_new_flaky", False)
    failed_ci_jobs = d.get("failed_ci_jobs", [])
    untracked_failed = d.get("untracked_failed_ci_jobs", [])
    workflow_jobs = d.get("workflow_jobs", [])

    # Slack user or user-group to mention on new failures or new flaky tests.
    # Set CUOPT_SLACK_MENTION_ID to either:
    #   - a user ID (starts with U or W, e.g. U01ABCDEF) — pings the user
    #   - a user-group / subteam ID (starts with S, e.g. S01ABCDEF) — pings the group
    # The group's handle name (e.g. "cuopt-ci-team") will NOT ping; Slack
    # requires the subteam ID, formatted as <!subteam^...>. Empty disables.
    mention_id = os.environ.get("CUOPT_SLACK_MENTION_ID", "")
    if mention_id.startswith("S"):
        mention_tag = f"<!subteam^{mention_id}> "
    elif mention_id:
        mention_tag = f"<@{mention_id}> "
    else:
        mention_tag = ""

    total_jobs = jobs.get("total", 0)

    total_ci_jobs = len(workflow_jobs)
    passed_ci_count = sum(
        1 for j in workflow_jobs if j.get("conclusion") == "success"
    )

    # ==================================================================
    # MAIN MESSAGE (line 1) -- posted to channel, becomes thread parent
    # ==================================================================
    blocks = []

    # Identify which workflows have failures (from both CI jobs and matrix grid)
    failing_workflows = set()
    for j in failed_ci_jobs:
        failing_workflows.add(_job_prefix(j))
    for g in grid:
        if str(g.get("status", "")).startswith("failed"):
            failing_workflows.add(g.get("test_type", "unknown"))
    flaky_workflows = set()
    for g in grid:
        if g.get("status") == "flaky":
            flaky_workflows.add(g.get("test_type", "unknown"))

    has_failures = len(failing_workflows) > 0
    untracked_count = len(untracked_failed)

    if has_failures and has_new:
        emoji = ":rotating_light:"
        text = f"{len(failing_workflows)} workflow(s) with NEW failures"
        if has_new_flaky:
            text += " + NEW flaky tests"
        mention = mention_tag
    elif has_failures and untracked_count > 0:
        emoji = ":rotating_light:"
        text = (
            f"Recurring failures in {len(failing_workflows)} workflow(s)"
            f" + {untracked_count} CI job(s) failed (no test details)"
        )
        mention = mention_tag
    elif has_failures and has_new_flaky:
        emoji = ":x:"
        text = f"Recurring failures in {len(failing_workflows)} workflow(s) + NEW flaky tests"
        mention = mention_tag
    elif has_failures:
        emoji = ":x:"
        text = f"Recurring failures in {len(failing_workflows)} workflow(s)"
        mention = ""
    elif flaky_workflows and has_new_flaky:
        emoji = ":large_yellow_circle:"
        text = "All jobs passed but NEW flaky tests detected"
        mention = mention_tag
    elif flaky_workflows:
        emoji = ":large_yellow_circle:"
        text = "All jobs passed but flaky tests detected"
        mention = ""
    else:
        emoji = ":white_check_mark:"
        text = f"All {total_jobs} matrix jobs passed"
        if total_ci_jobs > 0:
            if passed_ci_count == total_ci_jobs:
                text += f", all {total_ci_jobs} CI jobs succeeded"
            else:
                text += (
                    f", {passed_ci_count}/{total_ci_jobs} CI jobs succeeded"
                )
        mention = ""

    stats_parts = []
    if totals.get("failed", 0) > 0:
        stats_parts.append(f":x: {totals['failed']} failed")
    if totals.get("flaky", 0) > 0:
        stats_parts.append(f":warning: {totals['flaky']} flaky")
    if not stats_parts:
        stats_parts.append(
            f":white_check_mark: {totals.get('passed', totals.get('total', 0))} tests passed"
        )
    if totals.get("skipped", 0) > 0:
        stats_parts.append(f":fast_forward: {totals['skipped']} skipped")
    stats = "  |  ".join(stats_parts)

    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"cuOpt Nightly Tests \u2014 {branch} \u2014 {date}",
                "emoji": True,
            },
        }
    )
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{mention}{emoji} *{_esc(text)}*\n\n{_esc(stats)}",
            },
        }
    )

    # Per-workflow failure summary using CI job counts from GitHub API
    # Build a lookup: workflow prefix -> (failed, total) from workflow_jobs
    wf_counts = {}
    for j in workflow_jobs:
        prefix = _job_prefix(j)
        wf_counts.setdefault(prefix, {"failed": 0, "total": 0})
        wf_counts[prefix]["total"] += 1
        if j.get("conclusion") == "failure":
            wf_counts[prefix]["failed"] += 1

    # Build a lookup: workflow prefix -> list of failing matrix_labels from grid
    wf_failing_labels = {}
    for g in grid:
        if str(g.get("status", "")).startswith("failed"):
            wf_failing_labels.setdefault(
                g.get("test_type", "unknown"), []
            ).append(g.get("matrix_label", "unknown"))

    if failing_workflows:
        lines = []
        for wf in sorted(failing_workflows):
            counts = wf_counts.get(wf, {})
            f_count = counts.get("failed", 0)
            t_count = counts.get("total", 0)
            # Append failing matrix labels (up to 3, then "+N more")
            labels = wf_failing_labels.get(wf, [])
            label_suffix = ""
            if labels:
                shown = labels[:3]
                label_suffix = " (" + ", ".join(shown)
                if len(labels) > 3:
                    label_suffix += f", +{len(labels) - 3} more"
                label_suffix += ")"
            if t_count > 0:
                lines.append(
                    f":x:  *{_esc(wf)}* \u2014 {f_count}/{t_count} failed{_esc(label_suffix)}"
                )
            else:
                lines.append(
                    f":x:  *{_esc(wf)}* \u2014 failed{_esc(label_suffix)}"
                )
        blocks.append({"type": "divider"})
        # Chunk to stay within Slack's 3000-char block limit
        current = ""
        for line in lines:
            if current and len(current) + len(line) + 1 > 2900:
                blocks.append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": current.rstrip()},
                    }
                )
                current = ""
            current += line + "\n"
        if current.strip():
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": current.rstrip()},
                }
            )

    # Commit under test, and what changed since the last green night
    sha = d.get("sha", "")
    last_green = d.get("last_green_sha", "")
    base = repo_base(github_run_url)
    commit_parts = []
    if sha and base:
        commit_parts.append(f"<{base}/commit/{sha}|`{sha[:7]}`>")
    elif sha:
        commit_parts.append(f"`{sha[:7]}`")
    if sha and last_green and last_green != sha and base and has_failures:
        commit_parts.append(
            f"<{base}/compare/{last_green}...{sha}|changes since last green"
            f" (`{last_green[:7]}`)>"
        )
    if commit_parts:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Commit " + "  |  ".join(commit_parts),
                    }
                ],
            }
        )

    # Links in main message
    link_parts = []
    if github_run_url:
        link_parts.append(f"<{github_run_url}|:github: GitHub Actions>")
    if presigned_report_url:
        link_parts.append(f"<{presigned_report_url}|:bar_chart: Full Report>")
    if presigned_dashboard_url:
        link_parts.append(
            f"<{presigned_dashboard_url}|:chart_with_upwards_trend: Dashboard>"
        )
    if link_parts:
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": "  |  ".join(link_parts)}
                ],
            }
        )

    print(make_payload(blocks))

    # ==================================================================
    # THREAD REPLIES (lines 2+) -- posted as replies to main message
    # ==================================================================

    # -- Thread 1: Failing and flaky tests (grouped by workflow) -------
    # Build per-workflow test issue lists
    new_failures = d.get("new_failures", [])
    recurring = d.get("recurring_failures", [])
    flaky = d.get("flaky_tests", [])
    resolved = d.get("resolved_tests", [])

    # Collect all test issues by test_type (workflow)
    issues_by_wf = {}
    for f_entry in new_failures:
        tt = f_entry.get("test_type", "unknown")
        issues_by_wf.setdefault(
            tt, {"new": [], "recurring": [], "flaky": [], "resolved": []}
        )
        issues_by_wf[tt]["new"].append(f_entry)
    for f_entry in recurring:
        tt = f_entry.get("test_type", "unknown")
        issues_by_wf.setdefault(
            tt, {"new": [], "recurring": [], "flaky": [], "resolved": []}
        )
        issues_by_wf[tt]["recurring"].append(f_entry)
    for f_entry in flaky:
        tt = f_entry.get("test_type", "unknown")
        issues_by_wf.setdefault(
            tt, {"new": [], "recurring": [], "flaky": [], "resolved": []}
        )
        issues_by_wf[tt]["flaky"].append(f_entry)
    for r in resolved:
        tt = r.get("test_type", "unknown")
        issues_by_wf.setdefault(
            tt, {"new": [], "recurring": [], "flaky": [], "resolved": []}
        )
        issues_by_wf[tt]["resolved"].append(r)

    def _since(e):
        return f" \u00b7 since {_esc(e.get('first_seen', '?'))}"

    def _reason(e):
        reason = clean_message(e.get("message", ""))
        return f" \u2014 {_esc(reason)}" if reason else ""

    for wf_name, issues in sorted(issues_by_wf.items()):
        wf_text = f"*{_esc(wf_name)}*\n"

        def _group_since(members):
            first = min(
                (m.get("first_seen") for m in members if m.get("first_seen")),
                default="?",
            )
            return f" \u00b7 since {_esc(first)}" + _group_nights(members)

        def _group_nights(members):
            n = max((m.get("nights_failed", 1) for m in members), default=1)
            return _nights({"nights_failed": n})

        sections = [
            (":new:", issues["new"], _reason, None, "new failures"),
            (
                lambda e: ":new: :warning:"
                if e.get("is_new")
                else ":warning:",
                issues["flaky"],
                lambda e: _reason(e) + _nights(e),
                _group_nights,
                "flaky",
            ),
            (
                ":repeat:",
                issues["recurring"],
                lambda e: _reason(e) + _since(e) + _nights(e),
                _group_since,
                "recurring",
            ),
        ]
        for tag, entries, detail, group_detail, label in sections:
            lines, hidden = issue_lines(entries, tag, detail, group_detail)
            for line in lines:
                wf_text += line + "\n"
            if hidden:
                wf_text += f"_...+{hidden} more {label}_\n"

        # Recovered tests are a count only; the names add noise.
        recovered = _dedupe(issues["resolved"])
        if recovered:
            if len(recovered) <= 3:
                names = ", ".join(
                    f"`{_esc(r.get('name', 'unknown'))}`" for r in recovered
                )
                wf_text += f":white_check_mark:  {len(recovered)} recovered: {names}\n"
            else:
                wf_text += (
                    f":white_check_mark:  {len(recovered)} tests recovered\n"
                )

        wf_blocks = []
        while wf_text:
            chunk = wf_text[:2900]
            wf_blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": chunk.rstrip()},
                }
            )
            wf_text = wf_text[2900:]

        print(make_payload(wf_blocks))

    # -- Thread: Failed job log links ----------------------------------
    failed_job_links = [
        j
        for j in workflow_jobs
        if j.get("conclusion") == "failure" and j.get("url")
    ]
    if failed_job_links:
        link_blocks = []
        current = "*Failed Job Logs:*\n"
        for j in failed_job_links:
            url = j.get("url", "")
            name = _esc(j.get("name", "unknown"))
            line = f":x:  <{url}|{name}>\n"
            if len(current) + len(line) > 2900:
                link_blocks.append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": current.rstrip()},
                    }
                )
                current = ""
            current += line
        if current.strip():
            link_blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": current.rstrip()},
                }
            )
        print(make_payload(link_blocks))


if __name__ == "__main__":
    main()
