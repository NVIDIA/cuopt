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

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
OUTPUT_DIR="${PWD}/aggregate-output"
mkdir -p "${OUTPUT_DIR}"

RUN_DATE="$(date +%F)"
BRANCH="${RAPIDS_BRANCH:-main}"

GITHUB_RUN_URL="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-NVIDIA/cuopt}/actions/runs/${GITHUB_RUN_ID:-}"

if [ -z "${CUOPT_DATASET_S3_URI:-}" ]; then
    echo "WARNING: CUOPT_DATASET_S3_URI is not set. Skipping nightly aggregation." >&2
    echo "The per-matrix reports (uploaded by individual test jobs) are still available on S3."
    exit 0
fi

S3_BASE="${CUOPT_DATASET_S3_URI}ci_test_reports/nightly"
S3_SUMMARIES_PREFIX="${S3_BASE}/summaries/${RUN_DATE}/"
S3_REPORTS_PREFIX="${S3_BASE}/reports/${RUN_DATE}/"
S3_CONSOLIDATED_JSON="${S3_BASE}/summaries/${RUN_DATE}/consolidated.json"
S3_CONSOLIDATED_HTML="${S3_BASE}/reports/${RUN_DATE}/consolidated.html"
S3_INDEX_URI="${S3_BASE}/index.json"
S3_DASHBOARD_URI="${S3_BASE}/dashboard/index.html"
DASHBOARD_DIR="${SCRIPT_DIR}/dashboard"

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
    --github-run-url "${GITHUB_RUN_URL}"

# Send consolidated Slack notification if webhook is available and this is a nightly build
if [ -n "${CUOPT_SLACK_WEBHOOK_URL:-}" ] && [ "${RAPIDS_BUILD_TYPE:-}" = "nightly" ]; then
    echo "Sending consolidated Slack notification"
    CONSOLIDATED_SUMMARY="${OUTPUT_DIR}/consolidated_summary.json" \
    CONSOLIDATED_HTML="${OUTPUT_DIR}/consolidated_report.html" \
    SLACK_WEBHOOK_URL="${CUOPT_SLACK_WEBHOOK_URL}" \
    SLACK_BOT_TOKEN="${CUOPT_SLACK_BOT_TOKEN:-}" \
    SLACK_CHANNEL_ID="${CUOPT_SLACK_CHANNEL_ID:-}" \
    REPORT_URL="${S3_CONSOLIDATED_HTML}" \
        bash "${SCRIPT_DIR}/utils/send_consolidated_summary.sh"
fi

echo "Nightly summary complete."
