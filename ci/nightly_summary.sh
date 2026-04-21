#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Aggregate all per-matrix nightly test summaries and send a single
# consolidated Slack notification.  Runs as a post-test job after all
# matrix CI jobs finish.
#
# The script needs S3 access via CUOPT_S3_URI (bucket root) and CUOPT_AWS_* credentials.
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

if [ -z "${CUOPT_S3_URI:-}" ]; then
    echo "WARNING: CUOPT_S3_URI is not set. Skipping nightly aggregation." >&2
    exit 0
fi

S3_BASE="${CUOPT_S3_URI}ci_test_reports/nightly"
BRANCH_SLUG=$(echo "${BRANCH}" | tr '/' '-')

# Summaries are scoped by GITHUB_RUN_ID so each workflow run is isolated.
# The run-scoped path has no date component — the run ID is unique, and
# dropping the date prevents mismatches when test jobs span midnight UTC.
# Fallback: branch-scoped path for backwards compat or non-CI runs.
if [ -n "${GITHUB_RUN_ID:-}" ]; then
    S3_SUMMARIES_PREFIX="${S3_BASE}/summaries/run-${GITHUB_RUN_ID}/"
else
    S3_SUMMARIES_PREFIX="${S3_BASE}/summaries/${RUN_DATE}/${BRANCH_SLUG}/"
fi
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


# Fallback: if the primary prefix is empty, try the branch-slug prefix.
# This handles cases where GITHUB_RUN_ID wasn't available in test containers
# (summaries were uploaded under the branch slug instead of run ID).
S3_SUMMARIES_FALLBACK="${S3_BASE}/summaries/${RUN_DATE}/${BRANCH_SLUG}/"

echo "Aggregating nightly summaries from ${S3_SUMMARIES_PREFIX}"

python3 "${SCRIPT_DIR}/utils/aggregate_nightly.py" \
    --s3-summaries-prefix "${S3_SUMMARIES_PREFIX}" \
    --s3-summaries-fallback "${S3_SUMMARIES_FALLBACK}" \
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

# --- Write GitHub Step Summary (if available) ---
if [ -n "${GITHUB_STEP_SUMMARY:-}" ] && [ -f "${OUTPUT_DIR}/consolidated_summary.json" ]; then
    python3 -c "
import json, sys

with open(sys.argv[1]) as f:
    d = json.load(f)
totals = d.get('test_totals', {})
grid = d.get('matrix_grid', [])
new_f = d.get('new_failures', [])
recur = d.get('recurring_failures', [])
flaky = d.get('flaky_tests', [])
resolved = d.get('resolved_tests', [])

print('# Nightly Test Summary — %s — %s' % (d.get('branch',''), d.get('date','')))
print()
print('| Metric | Count |')
print('|--------|-------|')
print('| Total | %d |' % totals.get('total', 0))
print('| Passed | %d |' % totals.get('passed', 0))
print('| **Failed** | **%d** |' % totals.get('failed', 0))
print('| Flaky | %d |' % totals.get('flaky', 0))
print('| Skipped | %d |' % totals.get('skipped', 0))
print('| Stabilized | %d |' % totals.get('resolved', 0))
print()
if new_f:
    print('## New Failures')
    print('| Test Type | Matrix | Test | Error |')
    print('|-----------|--------|------|-------|')
    for e in new_f[:20]:
        msg = (e.get('message','')[:80]).replace('\n',' ').replace('|','\\\\|')
        print('| %s | %s | \`%s\` | %s |' % (e.get('test_type',''), e.get('matrix_label',''), e['name'], msg))
    print()
if flaky:
    print('## Flaky Tests')
    print('| Test Type | Matrix | Test | Retries |')
    print('|-----------|--------|------|---------|')
    for e in flaky[:20]:
        print('| %s | %s | \`%s\` | %s |' % (e.get('test_type',''), e.get('matrix_label',''), e['name'], e.get('retry_count','?')))
    print()
if recur:
    print('## Recurring Failures')
    print('| Test Type | Matrix | Test | Since |')
    print('|-----------|--------|------|-------|')
    for e in recur[:20]:
        print('| %s | %s | \`%s\` | %s |' % (e.get('test_type',''), e.get('matrix_label',''), e['name'], e.get('first_seen','?')))
    print()
if resolved:
    print('## Stabilized Tests')
    for e in resolved[:10]:
        print('- \`%s\` (%s) — was failing %sx' % (e['name'], e.get('matrix_label',''), e.get('failure_count','?')))
    print()
print('## Matrix Overview')
print('| Test Type | Matrix | Status | Passed | Failed | Flaky |')
print('|-----------|--------|--------|--------|--------|-------|')
for g in grid:
    c = g.get('counts', {})
    print('| %s | %s | %s | %d | %d | %d |' % (g['test_type'], g['matrix_label'], g['status'], c.get('passed',0), c.get('failed',0), c.get('flaky',0)))
" "${OUTPUT_DIR}/consolidated_summary.json" >> "${GITHUB_STEP_SUMMARY}" || true
fi

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
