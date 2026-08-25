#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared helper for generating nightly test reports with matrix-aware S3 paths.
#
# Usage (source from any test script):
#
#   # For C++ tests (no Python version in matrix label):
#   generate_nightly_report "cpp"
#
#   # For Python tests (includes Python version in matrix label):
#   generate_nightly_report "python" --with-python-version
#
#   # For wheel tests:
#   generate_nightly_report "wheel-python" --with-python-version
#
# Prerequisites (set before calling):
#   RAPIDS_TESTS_DIR   - directory containing JUnit XML test results
#
# Optional environment variables (auto-detected if not set):
#   RAPIDS_CUDA_VERSION   - CUDA version (e.g., "12.9")
#   RAPIDS_PY_VERSION     - Python version (e.g., "3.12"), used with --with-python-version
#   RAPIDS_BRANCH         - branch name (e.g., "main")
#   RAPIDS_BUILD_TYPE     - build type; S3 history/summary/HTML uploads are
#                           only enabled when this equals "nightly"
#   CUOPT_S3_URI          - S3 bucket root (e.g., s3://cuopt-datasets/);
#                           only consulted when RAPIDS_BUILD_TYPE=nightly
#   GITHUB_SHA            - commit SHA
#   GITHUB_RUN_ID         - GitHub Actions run ID (scopes summaries to this run)
#   GITHUB_STEP_SUMMARY   - path for GitHub Actions step summary

# Resolve the directory where THIS helper lives (ci/utils/)
_HELPER_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

generate_nightly_report() {
    local test_type="${1:?Usage: generate_nightly_report <test_type> [--with-python-version]}"
    local include_py_version=false

    shift
    while [ $# -gt 0 ]; do
        case "$1" in
            --with-python-version) include_py_version=true ;;
            *) echo "WARNING: Unknown option: $1" >&2 ;;
        esac
        shift
    done

    # --- Build matrix label ---
    local cuda_tag="cuda${RAPIDS_CUDA_VERSION:-unknown}"
    local arch_tag
    arch_tag="$(arch)"
    local matrix_label="${cuda_tag}-${arch_tag}"

    if [ "${include_py_version}" = true ]; then
        local py_tag="py${RAPIDS_PY_VERSION:-unknown}"
        matrix_label="${cuda_tag}-${py_tag}-${arch_tag}"
    fi

    local branch_slug
    branch_slug=$(echo "${RAPIDS_BRANCH:-main}" | tr '/' '-')
    # Use RUN_DATE if set (nightly workflows pass the trigger date),
    # fall back to local date.  This avoids mismatches between test
    # jobs and the summary job when a run spans UTC midnight.
    local run_date
    run_date="${RUN_DATE:-$(date +%F)}"

    # --- Ensure results dir exists ---
    RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${PWD}/test-results}"
    mkdir -p "${RAPIDS_TESTS_DIR}"

    local report_output_dir="${RAPIDS_TESTS_DIR}/report"
    mkdir -p "${report_output_dir}"

    # --- Build S3 URIs ---
    local s3_history_uri=""
    local s3_history_seed_uri=""
    local s3_summary_uri=""
    local s3_summary_branch_uri=""
    local s3_html_uri=""

    # Only upload to S3 for nightly runs. For PRs and other build types we
    # still generate the local report and GitHub Step Summary, but skip S3
    # so PR runs don't pollute the nightly history/summary/report buckets.
    if [ "${RAPIDS_BUILD_TYPE:-}" = "nightly" ] && [ -n "${CUOPT_S3_URI:-}" ]; then
        local s3_base="${CUOPT_S3_URI}ci_test_reports/nightly"
        s3_history_uri="${s3_base}/history/${branch_slug}/${test_type}-${matrix_label}.json"
        # For non-main branches, seed history from main on first run so known
        # failures are inherited (not re-reported as new on release branches).
        if [ "${branch_slug}" != "main" ]; then
            s3_history_seed_uri="${s3_base}/history/main/${test_type}-${matrix_label}.json"
        fi
        # Scope summaries by GITHUB_RUN_ID so each workflow run is isolated.
        # The run-scoped path is date-free — the run ID is unique, and
        # dropping the date prevents mismatches when jobs span midnight UTC.
        # Also write to branch+date path for manual browsing.
        local summary_filename="${test_type}-${matrix_label}.json"
        if [ -n "${GITHUB_RUN_ID:-}" ]; then
            s3_summary_uri="${s3_base}/summaries/run-${GITHUB_RUN_ID}/${summary_filename}"
        else
            s3_summary_uri="${s3_base}/summaries/${run_date}/${branch_slug}/${summary_filename}"
        fi
        s3_summary_branch_uri="${s3_base}/summaries/${run_date}/${branch_slug}/${summary_filename}"
        s3_html_uri="${s3_base}/reports/${run_date}/${branch_slug}/${test_type}-${matrix_label}.html"
    fi

    # --- Run nightly report ---
    python3 "${_HELPER_DIR}/nightly_report.py" \
        --results-dir "${RAPIDS_TESTS_DIR}" \
        --output-dir "${report_output_dir}" \
        --sha "${GITHUB_SHA:-unknown}" \
        --date "${run_date}" \
        --test-type "${test_type}" \
        --matrix-label "${matrix_label}" \
        --s3-history-uri "${s3_history_uri}" \
        --s3-history-seed-uri "${s3_history_seed_uri}" \
        --s3-summary-uri "${s3_summary_uri}" \
        --s3-summary-branch-uri "${s3_summary_branch_uri}" \
        --s3-html-uri "${s3_html_uri}" \
        --github-step-summary "${GITHUB_STEP_SUMMARY:-}" \
        || true
}
