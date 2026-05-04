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
import sys


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

    # Slack user/group to mention on new failures or new flaky tests.
    # Set CUOPT_SLACK_MENTION_ID to a Slack user ID (e.g., U01ABCDEF) or
    # group handle. Empty disables mentions.
    mention_id = os.environ.get("CUOPT_SLACK_MENTION_ID", "")
    mention_tag = f"<@{mention_id}> " if mention_id else ""

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
            f":white_check_mark: {totals.get('total', 0)} tests passed"
        )
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

    if issues_by_wf:
        for wf_name, issues in sorted(issues_by_wf.items()):
            wf_blocks = []
            wf_text = f"*{_esc(wf_name)}*\n"

            # New failures first (most urgent, show more error context)
            for f_entry in issues["new"][:10]:
                msg = _esc(f_entry.get("message", "")[:150].replace("\n", " "))
                matrix = _esc(f_entry.get("matrix_label", ""))
                name = _esc(f_entry.get("name", "unknown"))
                wf_text += f":new:  `{name}` ({matrix}) \u2014 {msg}\n"

            # Flaky (actionable -- tests that are unstable)
            for f_entry in issues["flaky"][:10]:
                matrix = _esc(f_entry.get("matrix_label", ""))
                err = _esc(f_entry.get("message", "")[:100].replace("\n", " "))
                suffix = f" \u2014 {err}" if err else ""
                tag = (
                    ":new: :warning:" if f_entry.get("is_new") else ":warning:"
                )
                name = _esc(f_entry.get("name", "unknown"))
                wf_text += f"{tag}  `{name}` ({matrix}){suffix}\n"

            # Recurring failures (known issues)
            for f_entry in issues["recurring"][:10]:
                matrix = _esc(f_entry.get("matrix_label", ""))
                first = _esc(f_entry.get("first_seen", "?"))
                name = _esc(f_entry.get("name", "unknown"))
                wf_text += (
                    f":repeat:  `{name}` ({matrix}) \u2014 since {first}\n"
                )

            # Resolved
            for r in issues["resolved"][:5]:
                matrix = _esc(r.get("matrix_label", ""))
                count = r.get("failure_count", "?")
                name = _esc(r.get("name", "unknown"))
                wf_text += f":white_check_mark:  `{name}` ({matrix}) \u2014 was failing {count}x\n"

            # Truncation notes
            for category, label, limit in [
                ("new", "new failures", 10),
                ("recurring", "recurring", 10),
                ("flaky", "flaky", 10),
                ("resolved", "resolved", 5),
            ]:
                if len(issues[category]) > limit:
                    wf_text += (
                        f"_...+{len(issues[category]) - limit} more {label}_\n"
                    )

            # Chunk if needed
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
