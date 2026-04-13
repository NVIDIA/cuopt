#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Send a single consolidated Slack notification for the entire nightly run.
# Reads the aggregated JSON produced by aggregate_nightly.py and sends a rich
# Slack message with:
#   - Matrix grid overview (test_type x matrix → status)
#   - Failure tables with :new: / :repeat: badges and matrix context
#   - @channel on new genuine failures
#   - Stabilized and flaky test summaries
#   - Link to GitHub Actions run and consolidated HTML report
#
# Required environment variables:
#   SLACK_WEBHOOK_URL       - Slack incoming webhook URL
#   CONSOLIDATED_SUMMARY    - Path to consolidated_summary.json
#
# Optional environment variables:
#   REPORT_URL              - Link to the consolidated HTML report on S3
#   CONSOLIDATED_HTML       - Path to consolidated HTML file to upload to Slack
#   SLACK_BOT_TOKEN         - Slack Bot Token (xoxb-*) for file uploads
#   SLACK_CHANNEL_ID        - Slack channel ID for file uploads (required with bot token)

set -euo pipefail

CONSOLIDATED_SUMMARY="${CONSOLIDATED_SUMMARY:?CONSOLIDATED_SUMMARY must point to consolidated_summary.json}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:?SLACK_WEBHOOK_URL is required}"
REPORT_URL="${REPORT_URL:-}"
CONSOLIDATED_HTML="${CONSOLIDATED_HTML:-}"
SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-}"
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"

if [ ! -f "${CONSOLIDATED_SUMMARY}" ]; then
    echo "ERROR: Summary file not found: ${CONSOLIDATED_SUMMARY}" >&2
    exit 1
fi

PAYLOAD=$(python3 - "${CONSOLIDATED_SUMMARY}" "${REPORT_URL}" <<'PYEOF'
import json, sys

summary_path, report_url = sys.argv[1:3]

with open(summary_path) as f:
    d = json.load(f)

branch = d.get("branch", "main")
date = d.get("date", "unknown")
github_run_url = d.get("github_run_url", "")
jobs = d.get("job_summary", {})
totals = d.get("test_totals", {})
grid = d.get("matrix_grid", [])
has_new = d.get("has_new_failures", False)

total_jobs = jobs.get("total", 0)
failed_jobs = jobs.get("failed", 0)
flaky_jobs = jobs.get("flaky", 0)
passed_jobs = jobs.get("passed", 0)

# --- Status line ---
if failed_jobs > 0 and has_new:
    emoji = ":rotating_light:"
    text = f"NEW test failures in {failed_jobs} matrix job(s)"
    mention = "<!channel> "
elif failed_jobs > 0:
    emoji = ":x:"
    text = f"Recurring failures in {failed_jobs} matrix job(s)"
    mention = ""
elif flaky_jobs > 0:
    emoji = ":large_yellow_circle:"
    text = "All jobs passed but flaky tests detected"
    mention = ""
else:
    emoji = ":white_check_mark:"
    text = f"All {total_jobs} matrix jobs passed"
    mention = ""

stats = (
    f":white_check_mark: {totals.get('passed', 0)} passed  |  "
    f":x: {totals.get('failed', 0)} failed  |  "
    f":warning: {totals.get('flaky', 0)} flaky  |  "
    f":fast_forward: {totals.get('skipped', 0)} skipped  |  "
    f"Total: {totals.get('total', 0)}"
)

blocks = []

# Header
blocks.append({
    "type": "header",
    "text": {
        "type": "plain_text",
        "text": f"cuOpt Nightly Tests \u2014 {branch} \u2014 {date}",
        "emoji": True,
    },
})

# Status summary
blocks.append({
    "type": "section",
    "text": {
        "type": "mrkdwn",
        "text": f"{mention}{emoji} *{text}*\n\n{stats}",
    },
})

blocks.append({"type": "divider"})

# --- Matrix grid (compact) ---
# Group by test_type for readability
test_types = {}
for g in grid:
    tt = g["test_type"]
    test_types.setdefault(tt, []).append(g)

status_icons = {
    "passed": ":white_check_mark:",
    "failed-new": ":rotating_light:",
    "failed-recurring": ":x:",
    "flaky": ":warning:",
    "no-results": ":grey_question:",
}

grid_lines = []
for tt, entries in sorted(test_types.items()):
    cells = []
    for g in entries:
        icon = status_icons.get(g["status"], ":grey_question:")
        label = g["matrix_label"]
        failed_count = g["counts"].get("failed", 0)
        if failed_count > 0:
            cells.append(f"{icon} `{label}` ({failed_count} failures)")
        else:
            cells.append(f"{icon} `{label}`")
    grid_lines.append(f"*{tt}*\n" + "\n".join(f"    {c}" for c in cells))

# Slack blocks have a 3000 char limit per text field; truncate if needed
grid_text = "\n".join(grid_lines)
if len(grid_text) > 2900:
    # Summarize instead of full grid
    grid_text = (
        f"*Matrix Summary:* {passed_jobs} passed, {failed_jobs} failed, "
        f"{flaky_jobs} flaky out of {total_jobs} jobs\n"
        f"_(Full matrix in report link below)_"
    )

blocks.append({
    "type": "section",
    "text": {"type": "mrkdwn", "text": grid_text},
})

# --- New failures (max 10 to avoid hitting Slack limits) ---
new_failures = d.get("new_failures", [])
if new_failures:
    blocks.append({"type": "divider"})
    lines = []
    for f_entry in new_failures[:10]:
        msg = f_entry.get("message", "")[:50].replace("\n", " ")
        matrix = f_entry.get("matrix_label", "")
        lines.append(
            f"  :new:  `{f_entry['name']}` ({f_entry['test_type']} / {matrix}) \u2014 {msg}"
        )
    if len(new_failures) > 10:
        lines.append(f"  _...and {len(new_failures) - 10} more_")
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*New Failures:*\n" + "\n".join(lines)},
    })

# --- Recurring failures (max 10) ---
recurring = d.get("recurring_failures", [])
if recurring:
    blocks.append({"type": "divider"})
    lines = []
    for f_entry in recurring[:10]:
        matrix = f_entry.get("matrix_label", "")
        first = f_entry.get("first_seen", "?")
        lines.append(
            f"  :repeat:  `{f_entry['name']}` ({f_entry['test_type']} / {matrix}) \u2014 since {first}"
        )
    if len(recurring) > 10:
        lines.append(f"  _...and {len(recurring) - 10} more_")
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Recurring Failures:*\n" + "\n".join(lines)},
    })

# --- Stabilized ---
resolved = d.get("resolved_tests", [])
if resolved:
    lines = []
    for r in resolved[:5]:
        matrix = r.get("matrix_label", "")
        count = r.get("failure_count", "?")
        lines.append(
            f"  :white_check_mark:  `{r['name']}` ({r['test_type']} / {matrix}) \u2014 failed {count}x"
        )
    if len(resolved) > 5:
        lines.append(f"  _...and {len(resolved) - 5} more_")
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*Stabilized (were failing, now pass):*\n" + "\n".join(lines),
        },
    })

# --- Flaky summary (count only to save space) ---
flaky = d.get("flaky_tests", [])
if flaky:
    # Group by test name to show unique flaky tests
    unique_flaky = {}
    for f_entry in flaky:
        key = f_entry["name"]
        unique_flaky.setdefault(key, []).append(f_entry.get("matrix_label", ""))
    lines = []
    for name, matrices in sorted(unique_flaky.items())[:5]:
        matrix_str = ", ".join(matrices[:3])
        if len(matrices) > 3:
            matrix_str += f" +{len(matrices)-3} more"
        lines.append(f"  :warning:  `{name}` ({matrix_str})")
    if len(unique_flaky) > 5:
        lines.append(f"  _...and {len(unique_flaky) - 5} more unique flaky tests_")
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Flaky Tests:*\n" + "\n".join(lines)},
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

echo "Sending consolidated Slack notification..."
curl -s -X POST \
    -H 'Content-type: application/json' \
    --data "${PAYLOAD}" \
    "${SLACK_WEBHOOK_URL}"

echo ""
echo "Consolidated Slack notification sent."

# Upload HTML report as a file to Slack (requires bot token)
if [ -n "${SLACK_BOT_TOKEN}" ] && [ -n "${SLACK_CHANNEL_ID}" ] && [ -n "${CONSOLIDATED_HTML}" ] && [ -f "${CONSOLIDATED_HTML}" ]; then
    echo "Uploading HTML report to Slack..."

    # Read date and branch from the summary for the filename
    REPORT_DATE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('date','report'))" "${CONSOLIDATED_SUMMARY}" 2>/dev/null || echo "report")
    REPORT_BRANCH=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('branch','main'))" "${CONSOLIDATED_SUMMARY}" 2>/dev/null || echo "main")
    UPLOAD_FILENAME="cuopt-nightly-${REPORT_BRANCH}-${REPORT_DATE}.html"

    UPLOAD_RESPONSE=$(curl -s -X POST \
        -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
        -F "channels=${SLACK_CHANNEL_ID}" \
        -F "file=@${CONSOLIDATED_HTML}" \
        -F "filename=${UPLOAD_FILENAME}" \
        -F "title=cuOpt Nightly Report — ${REPORT_BRANCH} — ${REPORT_DATE}" \
        -F "initial_comment=Full nightly test report attached. Download and open in a browser for interactive details." \
        "https://slack.com/api/files.upload")

    if echo "${UPLOAD_RESPONSE}" | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('ok') else 1)" 2>/dev/null; then
        echo "HTML report uploaded to Slack."
    else
        echo "WARNING: Slack file upload failed. Response: ${UPLOAD_RESPONSE}" >&2
    fi
else
    if [ -n "${SLACK_BOT_TOKEN}" ] && [ -z "${SLACK_CHANNEL_ID}" ]; then
        echo "WARNING: SLACK_BOT_TOKEN set but SLACK_CHANNEL_ID missing, skipping file upload." >&2
    fi
fi
