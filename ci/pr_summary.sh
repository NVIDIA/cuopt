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
# Required env:
#   PR_NUMBER, GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID,
#   CUOPT_S3_URI, CUOPT_AWS_ACCESS_KEY_ID, CUOPT_AWS_SECRET_ACCESS_KEY
# Optional env:
#   GITHUB_BASE_REF (default: main), GITHUB_SERVER_URL, GITHUB_SHA

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
OUTPUT_DIR="${PWD}/pr-aggregate-output"
mkdir -p "${OUTPUT_DIR}"

: "${PR_NUMBER:?PR_NUMBER is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

if [ -z "${CUOPT_S3_URI:-}" ]; then
    echo "WARNING: CUOPT_S3_URI is not set; nothing to aggregate." >&2
    exit 0
fi
if [ -z "${GITHUB_RUN_ID:-}" ]; then
    echo "WARNING: GITHUB_RUN_ID is not set; cannot locate per-matrix summaries." >&2
    exit 0
fi

# aws CLI uses the standard AWS_* env vars; map the cuOpt-prefixed secrets onto them.
export AWS_ACCESS_KEY_ID="${CUOPT_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}"
export AWS_SECRET_ACCESS_KEY="${CUOPT_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
unset AWS_SESSION_TOKEN

GITHUB_RUN_URL="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
S3_PR_SUMMARIES_PREFIX="${CUOPT_S3_URI}ci_test_reports/pr/run-${GITHUB_RUN_ID}/"
COMMENT_FILE="${OUTPUT_DIR}/pr_comment.md"

echo "Aggregating PR per-matrix summaries from ${S3_PR_SUMMARIES_PREFIX}"
python3 "${SCRIPT_DIR}/utils/aggregate_pr.py" \
    --s3-pr-summaries-prefix "${S3_PR_SUMMARIES_PREFIX}" \
    --output-dir "${OUTPUT_DIR}" \
    --target-branch "${GITHUB_BASE_REF:-main}" \
    --sha "${GITHUB_SHA:-}" \
    --github-run-url "${GITHUB_RUN_URL}" \
    --run-date "$(date +%F)"

if [ ! -s "${COMMENT_FILE}" ]; then
    echo "No failures or flakes; not posting a PR comment."
    exit 0
fi

if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "ERROR: GITHUB_TOKEN is not set; cannot post PR comment." >&2
    echo "--- comment body that would have been posted ---" >&2
    cat "${COMMENT_FILE}" >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/utils/pr_comment_helper.py" post \
    --repo "${GITHUB_REPOSITORY}" \
    --pr "${PR_NUMBER}" \
    --body-file "${COMMENT_FILE}"
