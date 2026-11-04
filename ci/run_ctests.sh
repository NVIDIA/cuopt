#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run gtests through ctest, with per-test-case retry for flaky detection.
#
# Features:
#   - Selects tests by ctest label (CUOPT_CPP_TEST_GROUP)
#   - Collects JUnit XML results
#   - On failure, parses XML to find failing test cases and retries them individually
#   - Produces separate XML files per retry so nightly_report.py can classify flaky tests
#   - Detects segfaults (signal 11) and isolates crashing tests
#
# Environment variables:
#   GTEST_OUTPUT      - gtest XML output prefix (set by test_cpp.sh)
#   GTEST_MAX_RETRIES - max retries per failing test case (default: 2)
#   RAPIDS_TESTS_DIR  - directory for test results
#   CUOPT_CPP_TEST_GROUP - all (default), routing or mathopt
#
# Only --gtest_filter=<filter> is accepted as an argument.
#   CUOPT_CPP_TEST_GROUP - which binaries to run: all (default), routing
#                          (binaries labeled "routing" in cpp/tests) or mathopt
#                          (everything else, including unlabeled binaries)

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# shellcheck source=ci/utils/crash_helpers.sh
source "${SCRIPT_DIR}/utils/crash_helpers.sh"

# Support customizing the tests' install location
# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcuopt"
# Fall back to the build directory (devcontainer environments)
devcontainers_build_dir="${SCRIPT_DIR}/../cpp/build/latest"

if [[ -f "${installed_test_location}/CTestTestfile.cmake" ]]; then
    CTEST_DIR="${installed_test_location}"
    GTEST_DIR="${installed_test_location}"
elif [[ -f "${devcontainers_build_dir}/CTestTestfile.cmake" ]]; then
    CTEST_DIR="${devcontainers_build_dir}"
    GTEST_DIR="${devcontainers_build_dir}/gtests/libcuopt"
else
    echo "Error: CTestTestfile.cmake not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_build_dir}" >&2
    exit 1
fi

if ! command -v ctest > /dev/null; then
    echo "Error: ctest not found on PATH (install cmake in the test environment)" >&2
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
FAILED_BINARIES=()

# Record a failed gtest binary for the end-of-run summary.
# Args: <test_name> <reason>
record_binary_failure() {
    FAILED_BINARIES+=("$1 — $2")
}

# Synthesize a JUnit crash record so a binary-level crash is visible to
# nightly_report.py. gtest only writes its XML at the end of
# RUN_ALL_TESTS(); a SIGSEGV/SIGABRT mid-run leaves no XML behind, so
# without this record the failure is invisible to the classifier.
# Written to a separate *-crash.xml file to preserve any partial XML.
# Args: <test_name> <xml_dir> <rc>
write_binary_crash_marker() {
    local test_name="$1"
    local xml_dir="$2"
    local rc="$3"
    local sig
    sig=$(signal_name "${rc}")
    local crash_xml="${xml_dir}/${test_name}-crash.xml"
    write_crash_xml "${crash_xml}" "${test_name}" "PROCESS_CRASH" \
        "${test_name} crashed with ${sig} (exit code ${rc})" \
        "Process terminated by ${sig} mid-run. gtest did not emit a JUnit XML because RUN_ALL_TESTS() did not complete; inspect the run log for [FAILED] / stack-trace lines that preceded the crash."
}

# Run one test through ctest; returns the test's exit code.
# ctest reports only "Failed" for installed tests, so recover the code from its output.
# Args: <test_name> <xml_file> [gtest_filter]
run_ctest_test() {
    local test_name="$1"
    local xml_file="$2"
    local log
    log=$(mktemp)
    local -a filter_env=()
    [[ -n "${3:-}" ]] && filter_env=(GTEST_FILTER="$3")
    rm -f "${xml_file}"

    local ctest_rc=0
    set +o pipefail
    env ${filter_env[@]+"${filter_env[@]}"} GTEST_OUTPUT="xml:${xml_file}" ctest --test-dir "${CTEST_DIR}" \
        -R "^${test_name}\$" --no-tests=error --timeout 0 --verbose 2>&1 | tee "${log}"
    ctest_rc=${PIPESTATUS[0]}
    set -o pipefail

    local rc=0
    if [ "${ctest_rc}" -ne 0 ]; then
        if grep -Eiq "Segmentation fault|SegFault" "${log}"; then rc=139
        elif grep -Eiq "Subprocess aborted|Child aborted" "${log}"; then rc=134
        elif grep -Eiq "Illegal instruction|Exception: Illegal" "${log}"; then rc=132
        elif grep -Eiq "Bus error" "${log}"; then rc=135
        elif grep -Eiq "Floating-point exception|Exception: Numerical" "${log}"; then rc=136
        elif grep -Eiq "Subprocess killed" "${log}"; then rc=137
        elif grep -Eiq "Exception: Interrupt|Subprocess interrupted" "${log}"; then rc=130
        elif grep -Eq "\(Timeout\)|\*\*\*Timeout" "${log}"; then rc=124
        else
            rc=$(sed -n 's/.*Child return code: \([0-9]\+\).*/\1/p' "${log}" | tail -n 1)
            rc="${rc:-1}"
            [ "${rc}" -eq 0 ] && rc=1
        fi
    fi
    rm -f "${log}"
    return "${rc}"
}

run_gtest_with_retry() {
    local gt="$1"
    shift
    local test_name
    test_name=$(basename "${gt}")
    local xml_file="${RAPIDS_TESTS_DIR}/${test_name}.xml"

    echo "Running gtest ${test_name}"

    # First run — full binary, through CTest
    local rc=0
    run_ctest_test "${test_name}" "${xml_file}" || rc=$?

    if [ "${rc}" -eq 0 ]; then
        return 0
    fi

    # For non-nightly builds: fail immediately, no retries
    # PRs should surface failures directly so authors can see what broke
    if [ "${IS_NIGHTLY}" != "nightly" ]; then
        if was_signal_death "${rc}"; then
            local sig
            sig=$(signal_name "${rc}")
            echo "CRASH: ${test_name} died from ${sig} (exit code ${rc})"
            write_binary_crash_marker "${test_name}" "${RAPIDS_TESTS_DIR}" "${rc}"
            record_binary_failure "${test_name}" "CRASH (${sig})"
        else
            echo "FAILED: ${test_name} (exit code ${rc})"
            record_binary_failure "${test_name}" "exit ${rc}"
        fi
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
        all_tests=$("${gt}" --gtest_list_tests 2>/dev/null \
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
            record_binary_failure "${test_name}" "CRASH ($(signal_name ${rc})), gtest_list_tests unavailable"
            OVERALL_RC=1
            return 1
        fi
    else
        # Normal failure — extract which test cases failed from XML
        tests_to_retry=$(extract_failed_tests "${xml_file}")

        if [ -z "${tests_to_retry}" ]; then
            echo "FAILED: ${test_name} failed but could not identify failing test cases"
            record_binary_failure "${test_name}" "exit ${rc}, no failing testcase parseable from XML"
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
            run_ctest_test "${test_name}" "${retry_xml}" "${tc}" || retry_rc=$?

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
        record_binary_failure "${test_name}" "retries exhausted"
        OVERALL_RC=1
        return 1
    fi
    return 0
}

CUOPT_CPP_TEST_GROUP="${CUOPT_CPP_TEST_GROUP:-all}"
case "${CUOPT_CPP_TEST_GROUP}" in
    all | routing | mathopt) ;;
    *) echo "Error: invalid CUOPT_CPP_TEST_GROUP '${CUOPT_CPP_TEST_GROUP}'" >&2; exit 1 ;;
esac

for arg in "$@"; do
    case "${arg}" in
        --gtest_filter=*) export GTEST_FILTER="${arg#--gtest_filter=}" ;;
        *) echo "Error: unsupported argument '${arg}' (only --gtest_filter=... is supported)" >&2; exit 1 ;;
    esac
done

case "${CUOPT_CPP_TEST_GROUP}" in
    routing) CTEST_SELECT=(-L '^routing$') ;;
    mathopt) CTEST_SELECT=(-LE '^routing$') ;;
    *) CTEST_SELECT=() ;;
esac

# Unlabeled tests would silently land in mathopt, so require a group label.
UNLABELED=$(ctest --test-dir "${CTEST_DIR}" -N -LE '^(routing|numopt)$' |
    sed -n 's/^ *Test *#[0-9]*: //p' | grep -vx generate_resource_spec || true)
if [[ -n "${UNLABELED}" ]]; then
    echo "Error: tests without a routing or numopt label (set LABELS in CMake):" >&2
    echo "${UNLABELED}" | sed 's/^/  - /' >&2
    exit 1
fi

if ! TEST_LIST=$(ctest --test-dir "${CTEST_DIR}" -N "${CTEST_SELECT[@]}" --no-tests=error); then
    echo "Error: no tests selected for group '${CUOPT_CPP_TEST_GROUP}' in ${CTEST_DIR}" >&2
    exit 1
fi

NUM_RUN=0
while read -r test_name; do
    [[ -z "${test_name}" || "${test_name}" == generate_resource_spec ]] && continue
    NUM_RUN=$((NUM_RUN + 1))
    run_gtest_with_retry "${GTEST_DIR}/${test_name}" || true
done < <(sed -n 's/^ *Test *#[0-9]*: //p' <<< "${TEST_LIST}")

if [ "${NUM_RUN}" -eq 0 ]; then
    echo "Error: no tests selected for group '${CUOPT_CPP_TEST_GROUP}' in ${CTEST_DIR}" >&2
    OVERALL_RC=1
fi

# Final summary so failures are easy to spot in the raw run log.
# nightly_report.py also produces a structured report from the XML files,
# but this prints early (before any post-test-script steps) and surfaces
# crashes that bypassed gtest's XML output.
if [ "${#FAILED_BINARIES[@]}" -gt 0 ]; then
    echo ""
    echo "==================== FAILED gtest BINARIES (${#FAILED_BINARIES[@]}) ===================="
    for entry in "${FAILED_BINARIES[@]}"; do
        echo "  - ${entry}"
    done
    echo "================================================================"
fi

exit ${OVERALL_RC}
