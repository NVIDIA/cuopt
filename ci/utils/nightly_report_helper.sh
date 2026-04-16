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
#   CUOPT_DATASET_S3_URI  - S3 base URI for reports
#   GITHUB_SHA            - commit SHA
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
    local run_date
    run_date="$(date +%F)"

    # --- Ensure results dir exists ---
    RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${PWD}/test-results}"
    mkdir -p "${RAPIDS_TESTS_DIR}"

    local report_output_dir="${RAPIDS_TESTS_DIR}/report"
    mkdir -p "${report_output_dir}"

    # --- Build S3 URIs ---
    local s3_history_uri=""
    local s3_summary_uri=""
    local s3_html_uri=""

    if [ -n "${CUOPT_DATASET_S3_URI:-}" ]; then
        local s3_base="${CUOPT_DATASET_S3_URI}ci_test_reports/nightly"
        s3_history_uri="${s3_base}/history/${test_type}-${branch_slug}-${matrix_label}.json"
        s3_summary_uri="${s3_base}/summaries/${run_date}/${test_type}-${matrix_label}.json"
        s3_html_uri="${s3_base}/reports/${run_date}/${test_type}-${matrix_label}.html"
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
        --s3-summary-uri "${s3_summary_uri}" \
        --s3-html-uri "${s3_html_uri}" \
        --github-step-summary "${GITHUB_STEP_SUMMARY:-}" \
        || true
}
