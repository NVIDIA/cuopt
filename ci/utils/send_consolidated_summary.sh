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

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

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
PAYLOADS=$(python3 "${SCRIPT_DIR}/generate_slack_payloads.py" "${CONSOLIDATED_SUMMARY}" "${PRESIGNED_REPORT_URL}" "${PRESIGNED_DASHBOARD_URL}")

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
