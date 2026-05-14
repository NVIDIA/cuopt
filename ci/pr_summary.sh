#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregate per-matrix PR test summaries from S3 and post (or update) a
# sticky comment on the PR that classifies every failure as NEW
# (introduced by this PR) or KNOWN (recurring on nightly, known flaky on
# nightly, or flaked in this run via pytest-rerunfailures).
#
# Runs as a post-test job after every PR test job finishes.  See
# ci/utils/aggregate_pr.py for content generation and
# ci/utils/pr_comment_helper.py for GitHub API interactions.
#
# Required env (all must be set; the script exits with a loud error if any
# are missing):
#   PR_NUMBER, GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_BASE_REF,
#   GITHUB_SHA, GITHUB_TOKEN, CUOPT_S3_URI, CUOPT_AWS_ACCESS_KEY_ID,
#   CUOPT_AWS_SECRET_ACCESS_KEY.

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
