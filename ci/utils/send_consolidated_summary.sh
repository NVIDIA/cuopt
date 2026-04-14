#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Send a consolidated Slack notification for the entire nightly run.
# Reads the aggregated JSON produced by aggregate_nightly.py and sends
# chunked Slack messages:
#   1. Header + status summary + test totals + failed CI jobs
#   2. Failed/flaky matrix entries only (not passing ones)
#   3. Failure details (new, recurring, stabilized, flaky)
#   4. Links (presigned URLs + GitHub Actions)
# Then uploads the HTML report as a Slack file.
#
# Required environment variables:
#   SLACK_WEBHOOK_URL       - Slack incoming webhook URL
#   CONSOLIDATED_SUMMARY    - Path to consolidated_summary.json
#
# Optional environment variables:
#   CONSOLIDATED_HTML           - Path to consolidated HTML file to upload
#   SLACK_BOT_TOKEN             - Slack Bot Token (xoxb-*) for file uploads
#   SLACK_CHANNEL_ID            - Slack channel ID for file uploads
#   PRESIGNED_REPORT_URL        - Presigned URL for consolidated HTML report
#   PRESIGNED_DASHBOARD_URL     - Presigned URL for dashboard

set -euo pipefail

CONSOLIDATED_SUMMARY="${CONSOLIDATED_SUMMARY:?CONSOLIDATED_SUMMARY must point to consolidated_summary.json}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:?SLACK_WEBHOOK_URL is required}"
CONSOLIDATED_HTML="${CONSOLIDATED_HTML:-}"
SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-}"
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
PRESIGNED_REPORT_URL="${PRESIGNED_REPORT_URL:-}"
PRESIGNED_DASHBOARD_URL="${PRESIGNED_DASHBOARD_URL:-}"

if [ ! -f "${CONSOLIDATED_SUMMARY}" ]; then
    echo "ERROR: Summary file not found: ${CONSOLIDATED_SUMMARY}" >&2
    exit 1
fi

# Generate chunked Slack payloads — one JSON object per line
PAYLOADS=$(python3 - "${CONSOLIDATED_SUMMARY}" "${PRESIGNED_REPORT_URL}" "${PRESIGNED_DASHBOARD_URL}" <<'PYEOF'
import json, sys

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
failed_ci_jobs = d.get("failed_ci_jobs", [])
workflow_jobs = d.get("workflow_jobs", [])

total_jobs = jobs.get("total", 0)
failed_jobs = jobs.get("failed", 0)
flaky_jobs = jobs.get("flaky", 0)
passed_jobs = jobs.get("passed", 0)

# Count CI-level failures (jobs that failed at workflow level)
total_ci_jobs = len(workflow_jobs)
failed_ci_count = len(failed_ci_jobs)
passed_ci_count = sum(1 for j in workflow_jobs if j["conclusion"] == "success")

status_icons = {
    "passed": ":white_check_mark:",
    "failed-new": ":rotating_light:",
    "failed-recurring": ":x:",
    "flaky": ":warning:",
    "no-results": ":grey_question:",
}

def make_payload(blocks):
    return json.dumps({
        "username": "cuOpt Nightly Bot",
        "icon_emoji": ":robot_face:",
        "blocks": blocks,
    })


# ── Message 1: Header + status + totals + CI job failures ────────────
blocks = []

# Determine overall status considering both test results and CI jobs
all_green = failed_jobs == 0 and failed_ci_count == 0

if failed_ci_count > 0 or (failed_jobs > 0 and has_new):
    emoji = ":rotating_light:"
    parts = []
    if failed_ci_count > 0:
        parts.append(f"{failed_ci_count} CI job(s) failed")
    if failed_jobs > 0 and has_new:
        parts.append(f"NEW test failures in {failed_jobs} matrix job(s)")
    elif failed_jobs > 0:
        parts.append(f"recurring failures in {failed_jobs} matrix job(s)")
    text = " + ".join(parts)
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
    if total_ci_jobs > 0:
        text += f", all {passed_ci_count} CI jobs succeeded"
    mention = ""

stats = (
    f":white_check_mark: {totals.get('passed', 0)} passed  |  "
    f":x: {totals.get('failed', 0)} failed  |  "
    f":warning: {totals.get('flaky', 0)} flaky  |  "
    f":fast_forward: {totals.get('skipped', 0)} skipped  |  "
    f"Total: {totals.get('total', 0)}"
)

blocks.append({
    "type": "header",
    "text": {
        "type": "plain_text",
        "text": f"cuOpt Nightly Tests \u2014 {branch} \u2014 {date}",
        "emoji": True,
    },
})
blocks.append({
    "type": "section",
    "text": {
        "type": "mrkdwn",
        "text": f"{mention}{emoji} *{text}*\n\n{stats}",
    },
})

# Show failed CI jobs (notebooks, JuMP, etc.)
if failed_ci_jobs:
    lines = []
    for j in failed_ci_jobs:
        url = j.get("url", "")
        name = j["name"]
        if url:
            lines.append(f":x:  <{url}|{name}>")
        else:
            lines.append(f":x:  {name}")
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Failed CI Jobs:*\n" + "\n".join(lines)},
    })

print(make_payload(blocks))


# ── Message 2: Failed/flaky matrix entries only ──────────────────────
# Only show entries that are NOT passed
failed_grid = [g for g in grid if g["status"] != "passed"]

if failed_grid:
    test_types = {}
    for g in failed_grid:
        tt = g["test_type"]
        test_types.setdefault(tt, []).append(g)

    grid_blocks = []
    current_text = ""
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
        section = f"*{tt}*\n" + "\n".join(f"    {c}" for c in cells) + "\n"

        if current_text and len(current_text) + len(section) > 2800:
            grid_blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": current_text.rstrip()},
            })
            current_text = ""
        current_text += section

    if current_text:
        grid_blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": current_text.rstrip()},
        })

    for i in range(0, len(grid_blocks), 48):
        chunk = grid_blocks[i:i+48]
        print(make_payload([{"type": "divider"}] + chunk))
else:
    # All passed — just a compact summary
    if total_jobs > 0:
        print(make_payload([
            {"type": "divider"},
            {"type": "section",
             "text": {"type": "mrkdwn",
                      "text": f":white_check_mark: All {total_jobs} test matrix jobs passed"}},
        ]))


# ── Message 3: Failure details ────────────────────────────────────────
detail_blocks = []

# New failures
new_failures = d.get("new_failures", [])
if new_failures:
    lines = []
    for f_entry in new_failures[:15]:
        msg = f_entry.get("message", "")[:80].replace("\n", " ")
        matrix = f_entry.get("matrix_label", "")
        lines.append(
            f":new:  `{f_entry['name']}` ({f_entry['test_type']} / {matrix})\n       {msg}"
        )
    if len(new_failures) > 15:
        lines.append(f"_...and {len(new_failures) - 15} more_")
    text = "*:rotating_light: New Failures:*\n" + "\n".join(lines)
    while text:
        detail_blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": text[:2900]},
        })
        text = text[2900:]

# Recurring failures
recurring = d.get("recurring_failures", [])
if recurring:
    lines = []
    for f_entry in recurring[:15]:
        matrix = f_entry.get("matrix_label", "")
        first = f_entry.get("first_seen", "?")
        lines.append(
            f":repeat:  `{f_entry['name']}` ({f_entry['test_type']} / {matrix}) \u2014 since {first}"
        )
    if len(recurring) > 15:
        lines.append(f"_...and {len(recurring) - 15} more_")
    detail_blocks.append({"type": "divider"})
    detail_blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*:x: Recurring Failures:*\n" + "\n".join(lines)},
    })

# Stabilized
resolved = d.get("resolved_tests", [])
if resolved:
    lines = []
    for r in resolved[:10]:
        matrix = r.get("matrix_label", "")
        count = r.get("failure_count", "?")
        lines.append(
            f":white_check_mark:  `{r['name']}` ({r['test_type']} / {matrix}) \u2014 failed {count}x"
        )
    if len(resolved) > 10:
        lines.append(f"_...and {len(resolved) - 10} more_")
    detail_blocks.append({"type": "divider"})
    detail_blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "*:white_check_mark: Stabilized (were failing, now pass):*\n" + "\n".join(lines),
        },
    })

# Flaky summary
flaky = d.get("flaky_tests", [])
if flaky:
    unique_flaky = {}
    for f_entry in flaky:
        key = f_entry["name"]
        unique_flaky.setdefault(key, []).append(f_entry.get("matrix_label", ""))
    lines = []
    for name, matrices in sorted(unique_flaky.items())[:10]:
        matrix_str = ", ".join(matrices[:3])
        if len(matrices) > 3:
            matrix_str += f" +{len(matrices)-3} more"
        lines.append(f":warning:  `{name}` ({matrix_str})")
    if len(unique_flaky) > 10:
        lines.append(f"_...and {len(unique_flaky) - 10} more unique flaky tests_")
    detail_blocks.append({"type": "divider"})
    detail_blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*:warning: Flaky Tests:*\n" + "\n".join(lines)},
    })

if detail_blocks:
    print(make_payload(detail_blocks))


# ── Message 4: Links ─────────────────────────────────────────────────
link_parts = []
if github_run_url:
    link_parts.append(f"<{github_run_url}|:github: GitHub Actions>")
if presigned_report_url:
    link_parts.append(f"<{presigned_report_url}|:bar_chart: Full Report>")
if presigned_dashboard_url:
    link_parts.append(f"<{presigned_dashboard_url}|:chart_with_upwards_trend: Dashboard>")
if not presigned_report_url:
    link_parts.append("_Full report attached below_")

if link_parts:
    print(make_payload([
        {"type": "divider"},
        {"type": "context",
         "elements": [{"type": "mrkdwn", "text": "  |  ".join(link_parts)}]},
    ]))
PYEOF
)

echo "Sending consolidated Slack notification..."
while IFS= read -r payload; do
    response=$(curl -s -X POST \
        -H 'Content-type: application/json' \
        --data "${payload}" \
        "${SLACK_WEBHOOK_URL}")
    if [ "${response}" != "ok" ]; then
        echo "WARNING: Slack webhook returned: ${response}" >&2
    fi
done <<< "${PAYLOADS}"
echo "Consolidated Slack notification sent."

# Upload HTML report as a file to Slack (requires bot token)
if [ -n "${SLACK_BOT_TOKEN}" ] && [ -n "${SLACK_CHANNEL_ID}" ] && [ -n "${CONSOLIDATED_HTML}" ] && [ -f "${CONSOLIDATED_HTML}" ]; then
    echo "Uploading HTML report to Slack..."

    REPORT_DATE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('date','report'))" "${CONSOLIDATED_SUMMARY}" 2>/dev/null || echo "report")
    REPORT_BRANCH=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('branch','main'))" "${CONSOLIDATED_SUMMARY}" 2>/dev/null || echo "main")
    UPLOAD_FILENAME="cuopt-nightly-${REPORT_BRANCH}-${REPORT_DATE}.html"
    FILE_SIZE=$(stat --format=%s "${CONSOLIDATED_HTML}")
    UPLOAD_TITLE="cuOpt Nightly Report — ${REPORT_BRANCH} — ${REPORT_DATE}"

    # Step 1: Get an upload URL from Slack
    URL_RESPONSE=$(curl -s -X POST \
        -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        --data-urlencode "filename=${UPLOAD_FILENAME}" \
        --data-urlencode "length=${FILE_SIZE}" \
        "https://slack.com/api/files.getUploadURLExternal")

    UPLOAD_URL=$(echo "${URL_RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('upload_url',''))" 2>/dev/null)
    FILE_ID=$(echo "${URL_RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('file_id',''))" 2>/dev/null)

    if [ -z "${UPLOAD_URL}" ] || [ -z "${FILE_ID}" ]; then
        echo "WARNING: Slack file upload failed at getUploadURLExternal. Response: ${URL_RESPONSE}" >&2
    else
        # Step 2: Upload the file content to the presigned URL
        curl -s -X POST \
            -F "file=@${CONSOLIDATED_HTML}" \
            "${UPLOAD_URL}"

        # Step 3: Complete the upload and share to channel
        COMPLETE_PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({
    'files': [{'id': sys.argv[1], 'title': sys.argv[2]}],
    'channel_id': sys.argv[3],
    'initial_comment': 'Full nightly test report \u2014 download and open in a browser for interactive details.'
}))
" "${FILE_ID}" "${UPLOAD_TITLE}" "${SLACK_CHANNEL_ID}")

        COMPLETE_RESPONSE=$(curl -s -X POST \
            -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
            -H "Content-Type: application/json" \
            --data "${COMPLETE_PAYLOAD}" \
            "https://slack.com/api/files.completeUploadExternal")

        if echo "${COMPLETE_RESPONSE}" | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('ok') else 1)" 2>/dev/null; then
            echo "HTML report uploaded to Slack."
        else
            echo "WARNING: Slack file upload failed at completeUploadExternal. Response: ${COMPLETE_RESPONSE}" >&2
        fi
    fi
else
    if [ -n "${SLACK_BOT_TOKEN}" ] && [ -z "${SLACK_CHANNEL_ID}" ]; then
        echo "WARNING: SLACK_BOT_TOKEN set but SLACK_CHANNEL_ID missing, skipping file upload." >&2
    fi
fi
