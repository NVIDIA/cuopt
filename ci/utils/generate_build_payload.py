# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the Slack Block Kit payload for the build workflow summary.

Usage:
    python3 generate_build_payload.py <jobs.json> <branch> <date> <run_url> \
        [sha] [repo_url]

``jobs.json`` is the response of the GitHub "list jobs for a workflow run"
API. Prints one JSON payload to stdout.

A green build is one line. Only groups that need attention (failed,
cancelled, not finished) are listed. Skipped jobs are counted but not listed,
and they are not reported as passed.
"""

import json
import sys

# Jobs that are part of the workflow plumbing, not build results.
IGNORED = ("build-summary", "compute-matrix")


def conclusion(job):
    return job.get("conclusion") or "pending"


def group_name(job):
    name = job.get("name", "")
    return name.split(" / ")[0] if " / " in name else name


def build_payload(data, branch, date, run_url, sha="", repo_url=""):
    jobs = [
        j
        for j in data.get("jobs", [])
        if not any(i in j.get("name", "").lower() for i in IGNORED)
    ]

    counts = {}
    for j in jobs:
        counts[conclusion(j)] = counts.get(conclusion(j), 0) + 1
    failed = counts.get("failure", 0)
    cancelled = counts.get("cancelled", 0)
    skipped = counts.get("skipped", 0)
    passed = counts.get("success", 0)
    pending = counts.get("pending", 0)

    extras = []
    if skipped:
        extras.append(f"{skipped} skipped")
    if cancelled:
        extras.append(f"{cancelled} cancelled")
    if pending:
        extras.append(f"{pending} not finished")
    extra_text = f" ({', '.join(extras)})" if extras else ""

    if failed:
        emoji = ":x:"
        status = f"{failed} build job(s) failed, {passed} passed{extra_text}"
    elif cancelled or pending:
        emoji = ":no_entry_sign:"
        status = f"Build incomplete: {passed} passed{extra_text}"
    else:
        emoji = ":white_check_mark:"
        status = f"All {passed} build jobs passed" + (
            f" ({skipped} skipped)" if skipped else ""
        )

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"cuOpt Build — {branch} — {date}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"{emoji} *{status}*"},
        },
    ]

    groups = {}
    for j in jobs:
        groups.setdefault(group_name(j), []).append(j)

    lines = []
    for name, members in sorted(groups.items()):
        bad = [j for j in members if conclusion(j) == "failure"]
        stopped = [j for j in members if conclusion(j) == "cancelled"]
        waiting = [j for j in members if conclusion(j) == "pending"]
        if bad:
            line = f":x:  *{name}* — {len(bad)}/{len(members)} failed"
            if bad[0].get("html_url"):
                line += f"  <{bad[0]['html_url']}|View Logs>"
            lines.append(line)
        elif stopped:
            lines.append(
                f":no_entry_sign:  *{name}* — "
                f"{len(stopped)}/{len(members)} cancelled"
            )
        elif waiting:
            lines.append(
                f":hourglass_flowing_sand:  *{name}* — "
                f"{len(waiting)}/{len(members)} not finished"
            )

    if lines:
        blocks.append({"type": "divider"})
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

    parts = []
    if sha:
        label = f"`{sha[:7]}`"
        parts.append(
            f"Commit <{repo_url}/commit/{sha}|{label}>"
            if repo_url
            else f"Commit {label}"
        )
    if run_url:
        parts.append(f"<{run_url}|:github: GitHub Actions>")
    if parts:
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": "  |  ".join(parts)}],
            }
        )

    return {
        "username": "cuOpt Build Bot",
        "icon_emoji": ":package:",
        "blocks": blocks,
    }


def main():
    jobs_path, branch, date, run_url = sys.argv[1:5]
    sha = sys.argv[5] if len(sys.argv) > 5 else ""
    repo_url = sys.argv[6] if len(sys.argv) > 6 else ""
    with open(jobs_path) as f:
        data = json.load(f)
    print(
        json.dumps(build_payload(data, branch, date, run_url, sha, repo_url))
    )


if __name__ == "__main__":
    main()
