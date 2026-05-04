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
PAYLOAD=$(python3 -c "
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)
branch = sys.argv[2]
date = sys.argv[3]
run_url = sys.argv[4]

jobs = data.get('jobs', [])

# Filter out build-summary itself and compute-matrix helpers
jobs = [j for j in jobs
        if 'build-summary' not in j.get('name', '').lower()
        and 'compute-matrix' not in j.get('name', '').lower()]

# Group by workflow prefix
groups = {}
for j in jobs:
    name = j.get('name', '')
    prefix = name.split(' / ')[0] if ' / ' in name else name
    groups.setdefault(prefix, []).append(j)

total = len(jobs)
failed_count = sum(1 for j in jobs if j.get('conclusion') == 'failure')
passed_count = sum(1 for j in jobs if j.get('conclusion') == 'success')

if failed_count > 0:
    emoji = ':x:'
    status = f'{failed_count} build job(s) failed'
else:
    emoji = ':white_check_mark:'
    status = f'All {passed_count} build jobs passed'

blocks = []
blocks.append({
    'type': 'header',
    'text': {'type': 'plain_text', 'text': f'cuOpt Build \u2014 {branch} \u2014 {date}', 'emoji': True},
})
blocks.append({
    'type': 'section',
    'text': {'type': 'mrkdwn', 'text': f'{emoji} *{status}*'},
})
blocks.append({'type': 'divider'})

# Build status per group
lines = []
for group_name, group_jobs in sorted(groups.items()):
    g_passed = sum(1 for j in group_jobs if j.get('conclusion') == 'success')
    g_failed = sum(1 for j in group_jobs if j.get('conclusion') == 'failure')
    g_total = len(group_jobs)

    if g_failed > 0:
        icon = ':x:'
        detail = f'{g_failed}/{g_total} failed'
        # Add clickable log links for failed jobs
        failed_in_group = [j for j in group_jobs if j.get('conclusion') == 'failure']
        if failed_in_group and failed_in_group[0].get('html_url'):
            log_url = failed_in_group[0]['html_url']
            detail += f'  <{log_url}|View Logs>'
    elif g_passed == g_total:
        icon = ':white_check_mark:'
        detail = f'{g_total} passed'
    else:
        icon = ':grey_question:'
        detail = f'{g_passed}/{g_total} passed'
    lines.append(f'{icon}  *{group_name}* \u2014 {detail}')

current = ''
for line in lines:
    if current and len(current) + len(line) + 1 > 2900:
        blocks.append({'type': 'section', 'text': {'type': 'mrkdwn', 'text': current.rstrip()}})
        current = ''
    current += line + '\n'
if current.strip():
    blocks.append({'type': 'section', 'text': {'type': 'mrkdwn', 'text': current.rstrip()}})

# Link
if run_url:
    blocks.append({'type': 'divider'})
    blocks.append({
        'type': 'context',
        'elements': [{'type': 'mrkdwn', 'text': f'<{run_url}|:github: GitHub Actions>'}],
    })

print(json.dumps({
    'username': 'cuOpt Build Bot',
    'icon_emoji': ':package:',
    'blocks': blocks,
}))
" "${JOBS_FILE}" "${BRANCH}" "${RUN_DATE}" "${GITHUB_RUN_URL}")

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
