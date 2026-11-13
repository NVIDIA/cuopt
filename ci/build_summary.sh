#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Send a Slack notification summarizing the build workflow status.
# Queries the GitHub API for job statuses and posts a compact message.

set -euo pipefail

BRANCH="${RAPIDS_BRANCH:-main}"
RUN_DATE="$(date +%F)"
GITHUB_RUN_URL="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-NVIDIA/cuopt}/actions/runs/${GITHUB_RUN_ID:-}"
SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-}"
SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-}"
BUILD_SHA="${RAPIDS_SHA:-${GITHUB_SHA:-}}"
REPO_URL="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-NVIDIA/cuopt}"

if [ -z "${SLACK_BOT_TOKEN}" ] || [ -z "${SLACK_CHANNEL_ID}" ]; then
    echo "SLACK_BOT_TOKEN or SLACK_CHANNEL_ID not set, skipping build summary."
    exit 0
fi

# Fetch workflow job statuses
JOBS_FILE=$(mktemp)
if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "${GITHUB_RUN_ID:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
    echo "Fetching build job statuses from GitHub API..."
    curl -s -L --max-time 30 \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?per_page=100" \
        > "${JOBS_FILE}" || echo "{}" > "${JOBS_FILE}"
else
    echo "{}" > "${JOBS_FILE}"
fi

# Generate Slack payload
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
PAYLOAD=$(python3 "${SCRIPT_DIR}/utils/generate_build_payload.py" \
    "${JOBS_FILE}" "${BRANCH}" "${RUN_DATE}" "${GITHUB_RUN_URL}" \
    "${BUILD_SHA}" "${REPO_URL}")

rm -f "${JOBS_FILE}"

# Send via bot token
echo "Sending build summary to Slack..."
BOT_PAYLOAD=$(python3 -c "
import json, sys
p = json.loads(sys.argv[1])
p['channel'] = sys.argv[2]
print(json.dumps(p))
" "${PAYLOAD}" "${SLACK_CHANNEL_ID}")

RESPONSE=$(curl -s --max-time 30 -X POST \
    -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "${BOT_PAYLOAD}" \
    "https://slack.com/api/chat.postMessage" || echo '{"ok":false,"error":"curl_failed"}')

OK=$(echo "${RESPONSE}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ok',''))" 2>/dev/null || echo "")
if [ "${OK}" != "True" ]; then
    echo "ERROR: chat.postMessage failed: ${RESPONSE}" >&2
else
    echo "Build summary posted to Slack."
fi
