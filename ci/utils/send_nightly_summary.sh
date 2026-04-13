#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Enhanced Slack notification for nightly test results.
# Reads the JSON summary produced by nightly_report.py and sends a rich
# Slack message with:
#   - Failure tables with :new: / :repeat: badges
#   - @channel on new genuine failures
#   - Stabilized tests (were failing, now passing)
#   - Flaky test list
#
# Required environment variables:
#   SLACK_WEBHOOK_URL  - Slack incoming webhook URL (set from CUOPT_SLACK_WEBHOOK_URL in CI)
#   NIGHTLY_SUMMARY    - Path to nightly_summary.json from nightly_report.py
#
# Optional environment variables:
#   GITHUB_RUN_URL     - Link to the GitHub Actions run
#   REPORT_URL         - Link to the S3 HTML report
#   CUOPT_BRANCH       - Branch name (e.g. main, release/26.06)

set -euo pipefail

NIGHTLY_SUMMARY="${NIGHTLY_SUMMARY:?NIGHTLY_SUMMARY must point to nightly_summary.json}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:?SLACK_WEBHOOK_URL is required}"
GITHUB_RUN_URL="${GITHUB_RUN_URL:-}"
REPORT_URL="${REPORT_URL:-}"
CUOPT_BRANCH="${CUOPT_BRANCH:-main}"

if [ ! -f "${NIGHTLY_SUMMARY}" ]; then
    echo "ERROR: Summary file not found: ${NIGHTLY_SUMMARY}" >&2
    exit 1
fi

# Build the entire Slack payload in Python for safe JSON handling.
# Shell variable interpolation into nested JSON is brittle; Python reads the
# summary file directly and produces a valid JSON payload on stdout.
PAYLOAD=$(python3 - "${NIGHTLY_SUMMARY}" "${CUOPT_BRANCH}" "${GITHUB_RUN_URL}" "${REPORT_URL}" <<'PYEOF'
import json, sys

summary_path, branch, github_run_url, report_url = sys.argv[1:5]

with open(summary_path) as f:
    d = json.load(f)

counts = d["counts"]
total = counts["total"]
passed = counts["passed"]
failed = counts["failed"]
flaky = counts["flaky"]
skipped = counts["skipped"]
resolved = counts.get("resolved", 0)
has_new = d["has_new_failures"]

# --- Status line ---
if failed > 0:
    if has_new:
        emoji = ":rotating_light:"
        text = "NEW test failures detected"
        mention = "<!channel> "
    else:
        emoji = ":x:"
        text = "Recurring test failures"
        mention = ""
elif flaky > 0:
    emoji = ":large_yellow_circle:"
    text = "All passed but flaky tests detected"
    mention = ""
else:
    emoji = ":white_check_mark:"
    text = "All tests passed"
    mention = ""

stats = (
    f":white_check_mark: {passed} passed  |  :x: {failed} failed  |  "
    f":warning: {flaky} flaky  |  :fast_forward: {skipped} skipped  |  Total: {total}"
)

blocks = []

# Header
blocks.append({
    "type": "header",
    "text": {"type": "plain_text", "text": f"cuOpt Nightly Tests \u2014 {branch}", "emoji": True},
})

# Status summary
blocks.append({
    "type": "section",
    "text": {"type": "mrkdwn", "text": f"{mention}{emoji} *{text}*\n\n{stats}"},
})

blocks.append({"type": "divider"})

# --- Genuine failures ---
if failed > 0:
    lines = []
    for f_entry in d.get("new_failures", []):
        msg = f_entry.get("message", "")[:60].replace("\n", " ")
        lines.append(f"  :new:  `{f_entry['name']}` ({f_entry['suite']}) \u2014 {msg}")
    for f_entry in d.get("recurring_failures", []):
        msg = f_entry.get("message", "")[:60].replace("\n", " ")
        first = f_entry.get("first_seen", "?")
        lines.append(f"  :repeat:  `{f_entry['name']}` ({f_entry['suite']}) \u2014 since {first}")
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Genuine Failures:*\n" + "\n".join(lines)},
    })

# --- Stabilized tests ---
resolved_list = d.get("resolved_tests", [])
if resolved_list:
    lines = []
    for r in resolved_list:
        since = r.get("first_seen", "?")
        count = r.get("failure_count", "?")
        flaky_tag = " (was flaky)" if r.get("was_flaky") else ""
        lines.append(
            f"  :white_check_mark:  `{r['name']}` ({r['suite']}) \u2014 "
            f"failing since {since}, failed {count}x{flaky_tag}"
        )
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*Stabilized (were failing, now pass):*\n" + "\n".join(lines),
        },
    })

# --- Flaky tests ---
flaky_list = d.get("flaky_tests", [])
if flaky_list:
    lines = []
    for f_entry in flaky_list:
        retries = f_entry.get("retry_count", "?")
        lines.append(f"  :warning:  `{f_entry['name']}` ({f_entry['suite']}) \u2014 {retries} retries")
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Flaky Tests (passed on retry):*\n" + "\n".join(lines)},
    })

# --- Links ---
link_parts = []
if github_run_url:
    link_parts.append(f"<{github_run_url}|GitHub Actions>")
if report_url:
    link_parts.append(f"<{report_url}|Full Report>")
if link_parts:
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "  ".join(link_parts)}],
    })

payload = {
    "channel": "cuopt-regression-testing",
    "username": "cuOpt Nightly Bot",
    "icon_emoji": ":robot_face:",
    "blocks": blocks,
}
print(json.dumps(payload))
PYEOF
)

echo "Sending Slack notification..."
curl -s -X POST \
    -H 'Content-type: application/json' \
    --data "${PAYLOAD}" \
    "${SLACK_WEBHOOK_URL}"

echo ""
echo "Slack notification sent."
