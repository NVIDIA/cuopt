#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregate per-matrix PR test summaries from S3 and post (or update)
# the sticky PR classification comment.  See ci/utils/aggregate_pr.py
# (content) and ci/utils/pr_comment_helper.py (GitHub API).

set -euo pipefail

: "${PR_NUMBER:?required}"
: "${GITHUB_REPOSITORY:?required}"
: "${GITHUB_RUN_ID:?required}"
: "${GITHUB_BASE_REF:?required}"
: "${GITHUB_SHA:?required}"
: "${GITHUB_TOKEN:?required}"
: "${CUOPT_S3_URI:?required}"
: "${CUOPT_AWS_ACCESS_KEY_ID:?required}"
: "${CUOPT_AWS_SECRET_ACCESS_KEY:?required}"

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
OUTPUT_DIR="${PWD}/pr-aggregate-output"
mkdir -p "${OUTPUT_DIR}"

# aws CLI reads the standard AWS_* env vars; map the cuOpt-prefixed
# secrets onto them.
export AWS_ACCESS_KEY_ID="${CUOPT_AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${CUOPT_AWS_SECRET_ACCESS_KEY}"
unset AWS_SESSION_TOKEN

GITHUB_RUN_URL="https://github.com/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
S3_PR_SUMMARIES_PREFIX="${CUOPT_S3_URI}ci_test_reports/pr/run-${GITHUB_RUN_ID}/"
COMMENT_FILE="${OUTPUT_DIR}/pr_comment.md"

echo "Aggregating PR per-matrix summaries from ${S3_PR_SUMMARIES_PREFIX}"
python3 "${SCRIPT_DIR}/utils/aggregate_pr.py" \
    --s3-pr-summaries-prefix "${S3_PR_SUMMARIES_PREFIX}" \
    --output-dir "${OUTPUT_DIR}" \
    --target-branch "${GITHUB_BASE_REF}" \
    --sha "${GITHUB_SHA}" \
    --github-run-url "${GITHUB_RUN_URL}" \
    --run-date "$(date +%F)"

if [ ! -s "${COMMENT_FILE}" ]; then
    echo "No failures or flakes; not posting a PR comment."
    exit 0
fi

python3 "${SCRIPT_DIR}/utils/pr_comment_helper.py" post \
    --repo "${GITHUB_REPOSITORY}" \
    --pr "${PR_NUMBER}" \
    --body-file "${COMMENT_FILE}"
