#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run gtests with per-test-case retry for flaky detection.
#
# Features:
#   - Runs each gtest binary and collects JUnit XML results
#   - On failure, parses XML to find failing test cases and retries them individually
#   - Produces separate XML files per retry so nightly_report.py can classify flaky tests
#   - Detects segfaults (signal 11) and isolates crashing tests
#
# Environment variables:
#   GTEST_OUTPUT      - gtest XML output prefix (set by test_cpp.sh)
#   GTEST_MAX_RETRIES - max retries per failing test case (default: 2)
#   RAPIDS_TESTS_DIR  - directory for test results

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# shellcheck source=ci/utils/crash_helpers.sh
source "${SCRIPT_DIR}/utils/crash_helpers.sh"

# Support customizing the gtests' install location
# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcuopt/"
# Fall back to the build directory (devcontainer environments)
devcontainers_test_location="${SCRIPT_DIR}/../cpp/build/latest/gtests/libcuopt/"

if [[ -d "${installed_test_location}" ]]; then
    GTEST_DIR="${installed_test_location}"
elif [[ -d "${devcontainers_test_location}" ]]; then
    GTEST_DIR="${devcontainers_test_location}"
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_test_location}" >&2
    exit 1
fi

GTEST_MAX_RETRIES=${GTEST_MAX_RETRIES:-2}
RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${PWD}/test-results}"
IS_NIGHTLY="${RAPIDS_BUILD_TYPE:-}"


JUNIT_HELPERS="${SCRIPT_DIR}/utils/junit_helpers.py"

# Extract failing test case names from a gtest JUnit XML file
extract_failed_tests() {
    local xml_file="$1"
    if [ ! -f "${xml_file}" ]; then
        echo ""
        return
    fi
    python3 "${JUNIT_HELPERS}" failed "${xml_file}"
}

OVERALL_RC=0

run_gtest_with_retry() {
    local gt="$1"
    shift
    local test_name
    test_name=$(basename "${gt}")
    local xml_file="${RAPIDS_TESTS_DIR}/${test_name}.xml"

    echo "Running gtest ${test_name}"

    # First run — full binary
    local rc=0
    "${gt}" --gtest_output="xml:${xml_file}" "$@" || rc=$?

    if [ "${rc}" -eq 0 ]; then
        return 0
    fi

    # For non-nightly builds: fail immediately, no retries
    # PRs should surface failures directly so authors can see what broke
    if [ "${IS_NIGHTLY}" != "nightly" ]; then
        OVERALL_RC=1
        return 1
    fi

    # Determine which tests to retry
    local tests_to_retry=""

    if was_signal_death "${rc}"; then
        echo "CRASH: ${test_name} died from $(signal_name ${rc}) (exit code ${rc})"

        # Find tests that didn't get to run (not in the partial XML)
        # plus any that failed. Only retry those, not the ones that passed.
        echo "INFO: Finding tests that need retry in ${test_name}"
        local all_tests
        all_tests=$("${gt}" --gtest_list_tests "$@" 2>/dev/null \
            | python3 "${JUNIT_HELPERS}" gtest-list || echo "")

        # Extract tests that already passed from partial XML
        local passed_tests=""
        if [ -f "${xml_file}" ]; then
            passed_tests=$(python3 "${JUNIT_HELPERS}" passed "${xml_file}" || echo "")
        fi

        # Retry = all_tests - passed_tests
        if [ -n "${passed_tests}" ]; then
            tests_to_retry=$(comm -23 \
                <(echo "${all_tests}" | sort) \
                <(echo "${passed_tests}" | sort))
        else
            tests_to_retry="${all_tests}"
        fi

        if [ -z "${tests_to_retry}" ]; then
            echo "FAILED: Could not list tests in ${test_name}, cannot retry"
            write_crash_xml "${xml_file}" "${test_name}" "PROCESS_CRASH" \
                "${test_name} crashed with $(signal_name ${rc}) (exit code ${rc})" \
                "Process terminated by $(signal_name ${rc}). This may indicate a segfault, double-free, or stack overflow."
            OVERALL_RC=1
            return 1
        fi
    else
        # Normal failure — extract which test cases failed from XML
        tests_to_retry=$(extract_failed_tests "${xml_file}")

        if [ -z "${tests_to_retry}" ]; then
            echo "FAILED: ${test_name} failed but could not identify failing test cases"
            OVERALL_RC=1
            return 1
        fi
    fi

    local num_to_retry
    num_to_retry=$(echo "${tests_to_retry}" | wc -l)
    echo "INFO: Retrying ${num_to_retry} test case(s) from ${test_name} individually"

    # Retry each test case individually
    local all_passed=true
    while IFS= read -r tc; do
        local tc_passed=false
        for attempt in $(seq 1 "${GTEST_MAX_RETRIES}"); do
            local tc_safe
            tc_safe=$(echo "${tc}" | tr -c '[:alnum:]._-' '_')
            local retry_xml="${RAPIDS_TESTS_DIR}/${test_name}-retry${attempt}-${tc_safe}.xml"
            echo "  Retry ${attempt}/${GTEST_MAX_RETRIES}: ${tc}"

            local retry_rc=0
            "${gt}" --gtest_filter="${tc}" --gtest_output="xml:${retry_xml}" "$@" || retry_rc=$?

            if [ "${retry_rc}" -eq 0 ]; then
                echo "  FLAKY: ${tc} passed on retry ${attempt}"
                tc_passed=true
                break
            fi

            if was_signal_death "${retry_rc}"; then
                echo "  CRASH: ${tc} died from $(signal_name ${retry_rc}) on retry ${attempt}"
                write_crash_xml "${retry_xml}" "${test_name}" "${tc}" \
                    "${tc} crashed with $(signal_name ${retry_rc}) on retry ${attempt}" \
                    "Process terminated by $(signal_name ${retry_rc}). This test causes intermittent crashes."
                # Don't break — keep retrying, might be a flaky crash
            fi
        done

        if [ "${tc_passed}" = false ]; then
            echo "  FAILED: ${tc} failed after $((GTEST_MAX_RETRIES + 1)) attempts"
            all_passed=false
        fi
    done <<< "${tests_to_retry}"

    if [ "${all_passed}" = false ]; then
        OVERALL_RC=1
        return 1
    fi
    return 0
}

for gt in "${GTEST_DIR}"/*_TEST; do
    run_gtest_with_retry "${gt}" "$@" || true
done

# Run C_API_TEST with CPU memory for local solves (excluding time limit tests)
if [ -x "${GTEST_DIR}/C_API_TEST" ]; then
  echo "Running gtest C_API_TEST with CUOPT_USE_CPU_MEM_FOR_LOCAL"
  CUOPT_USE_CPU_MEM_FOR_LOCAL=1 run_gtest_with_retry "${GTEST_DIR}/C_API_TEST" --gtest_filter=-c_api/TimeLimitTestFixture.* "$@" || true
else
  echo "Skipping C_API_TEST with CUOPT_USE_CPU_MEM_FOR_LOCAL (binary not found)"
fi

exit ${OVERALL_RC}
