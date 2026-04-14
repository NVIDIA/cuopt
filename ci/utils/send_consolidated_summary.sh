#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Send a consolidated Slack notification for the entire nightly run.
# Reads the aggregated JSON produced by aggregate_nightly.py and sends:
#   - Main message: Header + status summary + test totals + failed CI jobs
#   - Thread replies: matrix details, failure details, links, HTML report
#
# If SLACK_BOT_TOKEN is available, posts via chat.postMessage (enables
# threading). Falls back to webhook (no threading) otherwise.
#
# Required environment variables:
#   SLACK_WEBHOOK_URL       - Slack incoming webhook URL (fallback)
#   CONSOLIDATED_SUMMARY    - Path to consolidated_summary.json
#
# Optional environment variables:
#   CONSOLIDATED_HTML           - Path to consolidated HTML file to upload
#   SLACK_BOT_TOKEN             - Slack Bot Token (xoxb-*) for threading + file uploads
#   SLACK_CHANNEL_ID            - Slack channel ID (required with bot token)
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

# Generate Slack payloads — one JSON object per line.
# Line 1 = main message, lines 2+ = thread replies.
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
untracked_failed = d.get("untracked_failed_ci_jobs", [])
workflow_jobs = d.get("workflow_jobs", [])

total_jobs = jobs.get("total", 0)
failed_jobs = jobs.get("failed", 0)
flaky_jobs = jobs.get("flaky", 0)
passed_jobs = jobs.get("passed", 0)

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


# ══════════════════════════════════════════════════════════════════════
# MAIN MESSAGE (line 1) — posted to channel, becomes thread parent
# ══════════════════════════════════════════════════════════════════════
blocks = []

untracked_count = len(untracked_failed)
if untracked_count > 0 or (failed_jobs > 0 and has_new):
    emoji = ":rotating_light:"
    parts = []
    if untracked_count > 0:
        names = [j["name"] for j in untracked_failed]
        parts.append(f"{untracked_count} CI job(s) failed ({', '.join(names[:3])})")
    if failed_jobs > 0 and has_new:
        parts.append(f"NEW test failures in {failed_jobs} matrix job(s)")
    elif failed_jobs > 0:
        parts.append(f"recurring failures in {failed_jobs} matrix job(s)")
    text = " + ".join(parts)
    mention = ""
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

# Failed untracked CI jobs in main message (details in thread)
if untracked_failed:
    names = [j["name"] for j in untracked_failed]
    summary = f":x: *{len(untracked_failed)} CI job(s) failed:* " + ", ".join(f"`{n}`" for n in names[:5])
    if len(names) > 5:
        summary += f" _+{len(names) - 5} more_"
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": summary},
    })

# Links in main message
link_parts = []
if github_run_url:
    link_parts.append(f"<{github_run_url}|:github: GitHub Actions>")
if presigned_report_url:
    link_parts.append(f"<{presigned_report_url}|:bar_chart: Full Report>")
if presigned_dashboard_url:
    link_parts.append(f"<{presigned_dashboard_url}|:chart_with_upwards_trend: Dashboard>")
if link_parts:
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "  |  ".join(link_parts)}],
    })

print(make_payload(blocks))


# ══════════════════════════════════════════════════════════════════════
# THREAD REPLIES (lines 2+) — posted as replies to main message
# ══════════════════════════════════════════════════════════════════════

# ── Thread 1: CI Workflow Status (all jobs) ───────────────────────────
# Shows every workflow job so new workflows are automatically visible.
if workflow_jobs:
    ci_icons = {"success": ":white_check_mark:", "failure": ":x:",
                "cancelled": ":no_entry_sign:", "skipped": ":fast_forward:"}

    # Group by workflow prefix (e.g., "conda-cpp-tests", "conda-notebook-tests")
    wf_groups = {}
    for j in workflow_jobs:
        # Use the part before " / " as group name, or full name
        prefix = j["name"].split(" / ")[0] if " / " in j["name"] else j["name"]
        wf_groups.setdefault(prefix, []).append(j)

    ci_blocks = []
    current = "*CI Workflow Status:*\n"
    for group_name, group_jobs in sorted(wf_groups.items()):
        passed = sum(1 for j in group_jobs if j["conclusion"] == "success")
        failed = sum(1 for j in group_jobs if j["conclusion"] == "failure")
        total = len(group_jobs)

        if failed > 0:
            icon = ":x:"
            detail = f"{failed}/{total} failed"
        elif passed == total:
            icon = ":white_check_mark:"
            detail = f"{total} passed"
        else:
            icon = ":grey_question:"
            detail = f"{passed}/{total} passed"

        line = f"{icon}  *{group_name}* — {detail}\n"
        if len(current) + len(line) > 2900:
            ci_blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": current.rstrip()},
            })
            current = ""
        current += line

    if current.strip():
        ci_blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": current.rstrip()},
        })
    print(make_payload(ci_blocks))

# ── Thread 2: Failing and flaky tests (grouped by workflow) ───────────
# Build per-workflow test issue lists
new_failures = d.get("new_failures", [])
recurring = d.get("recurring_failures", [])
flaky = d.get("flaky_tests", [])
resolved = d.get("resolved_tests", [])

# Collect all test issues by test_type (workflow)
issues_by_wf = {}
for f_entry in new_failures:
    tt = f_entry.get("test_type", "unknown")
    issues_by_wf.setdefault(tt, {"new": [], "recurring": [], "flaky": [], "resolved": []})
    issues_by_wf[tt]["new"].append(f_entry)
for f_entry in recurring:
    tt = f_entry.get("test_type", "unknown")
    issues_by_wf.setdefault(tt, {"new": [], "recurring": [], "flaky": [], "resolved": []})
    issues_by_wf[tt]["recurring"].append(f_entry)
for f_entry in flaky:
    tt = f_entry.get("test_type", "unknown")
    issues_by_wf.setdefault(tt, {"new": [], "recurring": [], "flaky": [], "resolved": []})
    issues_by_wf[tt]["flaky"].append(f_entry)
for r in resolved:
    tt = r.get("test_type", "unknown")
    issues_by_wf.setdefault(tt, {"new": [], "recurring": [], "flaky": [], "resolved": []})
    issues_by_wf[tt]["resolved"].append(r)

if issues_by_wf:
    for wf_name, issues in sorted(issues_by_wf.items()):
        wf_blocks = []
        wf_text = f"*{wf_name}*\n"

        # New failures
        for f_entry in issues["new"][:10]:
            msg = f_entry.get("message", "")[:60].replace("\n", " ")
            matrix = f_entry.get("matrix_label", "")
            wf_text += f":new:  `{f_entry['name']}` ({matrix}) — {msg}\n"

        # Recurring failures
        for f_entry in issues["recurring"][:10]:
            matrix = f_entry.get("matrix_label", "")
            first = f_entry.get("first_seen", "?")
            wf_text += f":repeat:  `{f_entry['name']}` ({matrix}) — since {first}\n"

        # Flaky
        for f_entry in issues["flaky"][:10]:
            matrix = f_entry.get("matrix_label", "")
            wf_text += f":warning:  `{f_entry['name']}` ({matrix})\n"

        # Resolved
        for r in issues["resolved"][:5]:
            matrix = r.get("matrix_label", "")
            count = r.get("failure_count", "?")
            wf_text += f":white_check_mark:  `{r['name']}` ({matrix}) — was failing {count}x\n"

        # Truncation notes
        for category, label, limit in [("new", "new failures", 10), ("recurring", "recurring", 10),
                                        ("flaky", "flaky", 10), ("resolved", "resolved", 5)]:
            if len(issues[category]) > limit:
                wf_text += f"_...+{len(issues[category]) - limit} more {label}_\n"

        # Chunk if needed
        while wf_text:
            chunk = wf_text[:2900]
            wf_blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": chunk.rstrip()},
            })
            wf_text = wf_text[2900:]

        print(make_payload(wf_blocks))
PYEOF
)

# ── Send messages ─────────────────────────────────────────────────────
echo "Sending consolidated Slack notification..."

THREAD_TS=""
FIRST=true

while IFS= read -r payload; do
    if [ "${FIRST}" = true ] && [ -n "${SLACK_BOT_TOKEN}" ] && [ -n "${SLACK_CHANNEL_ID}" ]; then
        # Post main message via chat.postMessage to get thread_ts
        BOT_PAYLOAD=$(python3 -c "
import json, sys
p = json.loads(sys.argv[1])
p['channel'] = sys.argv[2]
print(json.dumps(p))
" "${payload}" "${SLACK_CHANNEL_ID}")

        RESPONSE=$(curl -s -X POST \
            -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
            -H "Content-Type: application/json" \
            --data "${BOT_PAYLOAD}" \
            "https://slack.com/api/chat.postMessage")

        THREAD_TS=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ts',''))" 2>/dev/null || echo "")
        OK=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null || echo "")

        if [ "${OK}" != "True" ]; then
            echo "WARNING: chat.postMessage failed: ${RESPONSE}" >&2
            # Fall back to webhook for this and remaining messages
            THREAD_TS=""
            curl -s -X POST -H 'Content-type: application/json' --data "${payload}" "${SLACK_WEBHOOK_URL}" || true
        else
            echo "Main message posted (ts=${THREAD_TS})"
        fi
        FIRST=false
    elif [ -n "${THREAD_TS}" ] && [ -n "${SLACK_BOT_TOKEN}" ] && [ -n "${SLACK_CHANNEL_ID}" ]; then
        # Post thread reply via chat.postMessage
        THREAD_PAYLOAD=$(python3 -c "
import json, sys
p = json.loads(sys.argv[1])
p['channel'] = sys.argv[2]
p['thread_ts'] = sys.argv[3]
print(json.dumps(p))
" "${payload}" "${SLACK_CHANNEL_ID}" "${THREAD_TS}")

        RESPONSE=$(curl -s -X POST \
            -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
            -H "Content-Type: application/json" \
            --data "${THREAD_PAYLOAD}" \
            "https://slack.com/api/chat.postMessage")

        OK=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null || echo "")
        if [ "${OK}" != "True" ]; then
            echo "WARNING: Thread reply failed: ${RESPONSE}" >&2
        fi
    else
        # Fallback: webhook (no threading)
        response=$(curl -s -X POST \
            -H 'Content-type: application/json' \
            --data "${payload}" \
            "${SLACK_WEBHOOK_URL}")
        if [ "${response}" != "ok" ]; then
            echo "WARNING: Slack webhook returned: ${response}" >&2
        fi
        FIRST=false
    fi
done <<< "${PAYLOADS}"
echo "Consolidated Slack notification sent."

# ── Upload HTML report as file in thread ──────────────────────────────
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

        # Step 3: Complete the upload and share to channel (in thread if available)
        COMPLETE_PAYLOAD=$(python3 -c "
import json, sys
payload = {
    'files': [{'id': sys.argv[1], 'title': sys.argv[2]}],
    'channel_id': sys.argv[3],
    'initial_comment': 'Full nightly test report \u2014 download and open in a browser for interactive details.',
}
thread_ts = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else ''
if thread_ts:
    payload['thread_ts'] = thread_ts
print(json.dumps(payload))
" "${FILE_ID}" "${UPLOAD_TITLE}" "${SLACK_CHANNEL_ID}" "${THREAD_TS}")

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
