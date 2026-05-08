#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregate per-matrix PR test summaries and post (or update) a single
# sticky PR comment that classifies every failure as NEW (introduced by
# this PR) or KNOWN (recurring on nightly, known flaky, or flaked in this
# run only).
#
# Runs as a post-test job after every PR test job finishes.
#
# Required environment:
#   PR_NUMBER                     - PR number to comment on
#   GITHUB_TOKEN                  - token with pull-requests:write
#   GITHUB_REPOSITORY             - owner/repo (e.g., NVIDIA/cuopt)
#   GITHUB_RUN_ID                 - workflow run that produced summaries
#   CUOPT_S3_URI, CUOPT_AWS_*     - S3 bucket root + credentials
#
# Optional:
#   GITHUB_BASE_REF               - PR target branch (default: main)
#   GITHUB_SERVER_URL             - default: https://github.com

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
OUTPUT_DIR="${PWD}/pr-aggregate-output"
mkdir -p "${OUTPUT_DIR}"

COMMENT_MARKER="<!-- pr-test-classification -->"

if [ -z "${PR_NUMBER:-}" ]; then
    echo "ERROR: PR_NUMBER is not set; cannot post comment." >&2
    exit 1
fi
if [ -z "${GITHUB_REPOSITORY:-}" ]; then
    echo "ERROR: GITHUB_REPOSITORY is not set." >&2
    exit 1
fi

TARGET_BRANCH="${GITHUB_BASE_REF:-main}"
RUN_DATE=$(date +%F)
GITHUB_RUN_URL="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID:-}"

# Map CUOPT_AWS_* to standard AWS env vars for the aws CLI used by aggregate_pr.py
export AWS_ACCESS_KEY_ID="${CUOPT_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}"
export AWS_SECRET_ACCESS_KEY="${CUOPT_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
unset AWS_SESSION_TOKEN

if [ -z "${CUOPT_S3_URI:-}" ]; then
    echo "WARNING: CUOPT_S3_URI is not set; nothing to aggregate." >&2
    exit 0
fi
if [ -z "${GITHUB_RUN_ID:-}" ]; then
    echo "WARNING: GITHUB_RUN_ID is not set; cannot locate per-matrix summaries." >&2
    exit 0
fi

S3_PR_SUMMARIES_PREFIX="${CUOPT_S3_URI}ci_test_reports/pr/run-${GITHUB_RUN_ID}/"

echo "Aggregating PR per-matrix summaries from ${S3_PR_SUMMARIES_PREFIX}"

python3 "${SCRIPT_DIR}/utils/aggregate_pr.py" \
    --s3-pr-summaries-prefix "${S3_PR_SUMMARIES_PREFIX}" \
    --output-dir "${OUTPUT_DIR}" \
    --target-branch "${TARGET_BRANCH}" \
    --sha "${GITHUB_SHA:-}" \
    --github-run-url "${GITHUB_RUN_URL}" \
    --run-date "${RUN_DATE}"

COMMENT_FILE="${OUTPUT_DIR}/pr_comment.md"
if [ ! -s "${COMMENT_FILE}" ]; then
    echo "No failures or flakes; not posting a PR comment."
    exit 0
fi

if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "ERROR: GITHUB_TOKEN is not set; cannot post PR comment." >&2
    echo "--- comment body that would have been posted ---"
    cat "${COMMENT_FILE}"
    exit 1
fi

# --- Find an existing classification comment ---
EXISTING_ID=$(
    curl -s -L --max-time 30 \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments?per_page=100" \
        | python3 -c "
import json, sys
marker = ${COMMENT_MARKER@Q}
try:
    comments = json.load(sys.stdin)
except json.JSONDecodeError:
    sys.exit(0)
for c in comments:
    body = c.get('body') or ''
    if body.lstrip().startswith(marker):
        print(c['id'])
        break
"
)

# --- Build the JSON payload from the file (handles newlines/quotes safely) ---
PAYLOAD=$(python3 -c "
import json, sys
body = open('${COMMENT_FILE}').read()
print(json.dumps({'body': body}))
")

if [ -n "${EXISTING_ID}" ]; then
    echo "Updating existing PR comment ${EXISTING_ID}"
    curl -s -L --max-time 30 -X PATCH \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        -H "Content-Type: application/json" \
        -d "${PAYLOAD}" \
        "https://api.github.com/repos/${GITHUB_REPOSITORY}/issues/comments/${EXISTING_ID}" \
        > "${OUTPUT_DIR}/comment_response.json"
else
    echo "Creating new PR comment"
    curl -s -L --max-time 30 -X POST \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        -H "Content-Type: application/json" \
        -d "${PAYLOAD}" \
        "https://api.github.com/repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" \
        > "${OUTPUT_DIR}/comment_response.json"
fi

# Report the comment URL for the workflow log
python3 -c "
import json
try:
    d = json.load(open('${OUTPUT_DIR}/comment_response.json'))
except (json.JSONDecodeError, FileNotFoundError):
    print('WARNING: no comment response captured')
else:
    if 'html_url' in d:
        print(f\"PR comment posted: {d['html_url']}\")
    elif 'message' in d:
        import sys
        print(f\"ERROR: GitHub API returned {d.get('message')}\", file=sys.stderr)
        sys.exit(1)
"
