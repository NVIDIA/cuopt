#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Aggregate all per-matrix nightly test summaries and send a single
# consolidated Slack notification.  Runs as a post-test job after all
# matrix CI jobs finish.
#
# The script needs S3 access. It tries CUOPT_DATASET_S3_URI first, then
# falls back to standard AWS env vars set by aws-actions/configure-aws-credentials.
#
# Optional:
#   CUOPT_SLACK_WEBHOOK_URL       - sends Slack if set
#   RAPIDS_BRANCH                 - branch name (default: main)
#   RAPIDS_BUILD_TYPE             - build type (nightly, pull-request, etc.)
#   GITHUB_TOKEN                  - for querying workflow job statuses
#   GITHUB_RUN_ID                 - current workflow run ID

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
OUTPUT_DIR="${PWD}/aggregate-output"
mkdir -p "${OUTPUT_DIR}"

RUN_DATE="${RUN_DATE:-$(date +%F)}"
BRANCH="${RAPIDS_BRANCH:-main}"

GITHUB_RUN_URL="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-NVIDIA/cuopt}/actions/runs/${GITHUB_RUN_ID:-}"

# Map CUOPT_AWS_* to standard AWS env vars for the aws CLI
export AWS_ACCESS_KEY_ID="${CUOPT_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}"
export AWS_SECRET_ACCESS_KEY="${CUOPT_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
unset AWS_SESSION_TOKEN

if [ -z "${CUOPT_DATASET_S3_URI:-}" ]; then
    echo "WARNING: CUOPT_DATASET_S3_URI is not set. Skipping nightly aggregation." >&2
    echo "The per-matrix reports (uploaded by individual test jobs) are still available on S3."
    exit 0
fi

S3_BASE="${CUOPT_DATASET_S3_URI}ci_test_reports/nightly"
BRANCH_SLUG=$(echo "${BRANCH}" | tr '/' '-')
# Per-matrix summaries are uploaded by test jobs under summaries/{date}/{branch}/.
# For production nightlies (main, release/*), RAPIDS_BRANCH matches the branch input.
# For feature branch testing, RAPIDS_BRANCH may default to "main" in rapidsai containers,
# so we search the date prefix recursively (s3_list handles this).
S3_SUMMARIES_PREFIX="${S3_BASE}/summaries/${RUN_DATE}/${BRANCH_SLUG}/"
S3_REPORTS_PREFIX="${S3_BASE}/reports/${RUN_DATE}/${BRANCH_SLUG}/"
S3_CONSOLIDATED_JSON="${S3_BASE}/summaries/${RUN_DATE}/${BRANCH_SLUG}/consolidated.json"
S3_CONSOLIDATED_HTML="${S3_BASE}/reports/${RUN_DATE}/${BRANCH_SLUG}/consolidated.html"
S3_INDEX_URI="${S3_BASE}/index.json"
S3_DASHBOARD_URI="${S3_BASE}/dashboard/${BRANCH_SLUG}/index.html"
DASHBOARD_DIR="${SCRIPT_DIR}/dashboard"

# --- Query GitHub API for workflow job statuses ---
WORKFLOW_JOBS_JSON="${OUTPUT_DIR}/workflow_jobs.json"
if [ -n "${GITHUB_TOKEN:-}" ] && [ -n "${GITHUB_RUN_ID:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
    echo "Fetching workflow job statuses from GitHub API..."
    curl -s -L \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?per_page=100" \
        > "${WORKFLOW_JOBS_JSON}" || echo "{}" > "${WORKFLOW_JOBS_JSON}"
else
    echo "WARNING: GITHUB_TOKEN or GITHUB_RUN_ID not set, skipping workflow job status." >&2
    echo "{}" > "${WORKFLOW_JOBS_JSON}"
fi


echo "RUN_DATE=${RUN_DATE}, BRANCH=${BRANCH}, BRANCH_SLUG=${BRANCH_SLUG}"
echo "Listing S3 summaries at ${S3_SUMMARIES_PREFIX}:"
aws s3 ls "${S3_SUMMARIES_PREFIX}" 2>&1 || echo "(no files or access error)"
# Diagnostic: show what's on S3 for this date
echo "=== S3 diagnostics ==="
echo "RUN_DATE=${RUN_DATE} BRANCH=${BRANCH} BRANCH_SLUG=${BRANCH_SLUG}"
echo "Looking for summaries at: ${S3_SUMMARIES_PREFIX}"
aws s3 ls "${S3_SUMMARIES_PREFIX}" 2>&1 | head -5 || true
echo "All summaries for ${RUN_DATE}:"
aws s3 ls "${S3_BASE}/summaries/${RUN_DATE}/" 2>&1 | head -10 || true
echo "=== End diagnostics ==="

echo "Aggregating nightly summaries from ${S3_SUMMARIES_PREFIX}"

python3 "${SCRIPT_DIR}/utils/aggregate_nightly.py" \
    --s3-summaries-prefix "${S3_SUMMARIES_PREFIX}" \
    --s3-reports-prefix "${S3_REPORTS_PREFIX}" \
    --s3-output-uri "${S3_CONSOLIDATED_JSON}" \
    --s3-html-output-uri "${S3_CONSOLIDATED_HTML}" \
    --s3-index-uri "${S3_INDEX_URI}" \
    --s3-dashboard-uri "${S3_DASHBOARD_URI}" \
    --dashboard-dir "${DASHBOARD_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --date "${RUN_DATE}" \
    --branch "${BRANCH}" \
    --github-run-url "${GITHUB_RUN_URL}" \
    --workflow-jobs "${WORKFLOW_JOBS_JSON}"

# --- Generate presigned URLs for reports (7-day expiry) ---
PRESIGN_EXPIRY=604800
PRESIGNED_HTML=$(aws s3 presign "${S3_CONSOLIDATED_HTML}" --expires-in "${PRESIGN_EXPIRY}" 2>&1) || {
    echo "WARNING: Failed to generate presigned URL for report: ${PRESIGNED_HTML}" >&2
    PRESIGNED_HTML=""
}
PRESIGNED_DASHBOARD=$(aws s3 presign "${S3_DASHBOARD_URI}" --expires-in "${PRESIGN_EXPIRY}" 2>&1) || {
    echo "WARNING: Failed to generate presigned URL for dashboard: ${PRESIGNED_DASHBOARD}" >&2
    PRESIGNED_DASHBOARD=""
}

# Send consolidated Slack notification if webhook is available and this is a nightly build
if [ -n "${CUOPT_SLACK_WEBHOOK_URL:-}" ] && [ "${RAPIDS_BUILD_TYPE:-}" = "nightly" ]; then
    echo "Sending consolidated Slack notification"
    CONSOLIDATED_SUMMARY="${OUTPUT_DIR}/consolidated_summary.json" \
    CONSOLIDATED_HTML="${OUTPUT_DIR}/consolidated_report.html" \
    SLACK_WEBHOOK_URL="${CUOPT_SLACK_WEBHOOK_URL}" \
    SLACK_BOT_TOKEN="${CUOPT_SLACK_BOT_TOKEN:-}" \
    SLACK_CHANNEL_ID="${CUOPT_SLACK_CHANNEL_ID:-}" \
    PRESIGNED_REPORT_URL="${PRESIGNED_HTML}" \
    PRESIGNED_DASHBOARD_URL="${PRESIGNED_DASHBOARD}" \
        bash "${SCRIPT_DIR}/utils/send_consolidated_summary.sh"
fi

echo "Nightly summary complete."
