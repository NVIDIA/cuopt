#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run gtests with per-test-case retry for flaky detection and coredump collection.
#
# Features:
#   - Runs each gtest binary and collects JUnit XML results
#   - On failure, parses XML to find failing test cases and retries them individually
#   - Produces separate XML files per retry so nightly_report.py can classify flaky tests
#   - Detects segfaults (signal 11) and collects coredumps if available
#
# Environment variables:
#   GTEST_OUTPUT      - gtest XML output prefix (set by test_cpp.sh)
#   GTEST_MAX_RETRIES - max retries per failing test case (default: 2)
#   RAPIDS_TESTS_DIR  - directory for test results (for coredump collection)

set -euo pipefail

# Support customizing the gtests' install location
# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcuopt/"
# Fall back to the build directory (devcontainer environments)
devcontainers_test_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest/gtests/libcuopt/"

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
COREDUMP_DIR="${RAPIDS_TESTS_DIR}/coredumps"
mkdir -p "${COREDUMP_DIR}"

# Enable coredumps
ulimit -c unlimited 2>/dev/null || true
# Set coredump path if possible (requires root or sysctl access)
if [ -w /proc/sys/kernel/core_pattern ] 2>/dev/null; then
    echo "${COREDUMP_DIR}/core.%e.%p" > /proc/sys/kernel/core_pattern
fi

# Extract failing test case names from a gtest JUnit XML file
extract_failed_tests() {
    local xml_file="$1"
    if [ ! -f "${xml_file}" ]; then
        echo ""
        return
    fi
    # Parse XML for failed testcases: classname.name
    python3 -c "
import sys
from xml.etree import ElementTree
try:
    tree = ElementTree.parse(sys.argv[1])
    for tc in tree.iter('testcase'):
        if tc.find('failure') is not None or tc.find('error') is not None:
            cls = tc.get('classname', '')
            name = tc.get('name', '')
            if cls and name:
                print(f'{cls}.{name}')
except Exception:
    pass
" "${xml_file}"
}

# Check if process died from a signal (segfault, abort, etc.)
was_signal_death() {
    local exit_code="$1"
    # Exit codes > 128 indicate signal death: exit_code = 128 + signal_number
    # Signal 11 = SIGSEGV, Signal 6 = SIGABRT
    if [ "${exit_code}" -gt 128 ]; then
        return 0
    fi
    return 1
}

signal_name() {
    local exit_code="$1"
    local sig=$((exit_code - 128))
    case "${sig}" in
        6)  echo "SIGABRT" ;;
        11) echo "SIGSEGV (segfault)" ;;
        *)  echo "signal ${sig}" ;;
    esac
}

# Collect any coredumps generated after a test run
collect_coredumps() {
    local test_name="$1"
    local found=false
    # Check common coredump locations
    for pattern in "${COREDUMP_DIR}/core.*" "/tmp/core.*" "core" "core.*"; do
        for corefile in ${pattern}; do
            if [ -f "${corefile}" ]; then
                local dest="${COREDUMP_DIR}/${test_name}-$(basename "${corefile}")"
                mv "${corefile}" "${dest}" 2>/dev/null || true
                echo "COREDUMP: Collected ${dest}"
                found=true
            fi
        done
    done
    if [ "${found}" = true ]; then
        echo "COREDUMP: Core dumps saved to ${COREDUMP_DIR}/"
    fi
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

    # Check for signal death (segfault, abort)
    if was_signal_death "${rc}"; then
        echo "CRASH: ${test_name} died from $(signal_name ${rc}) (exit code ${rc})"
        collect_coredumps "${test_name}"
        # Write a minimal JUnit XML for the crash so it appears in reports
        if [ ! -f "${xml_file}" ] || ! grep -q 'testcase' "${xml_file}" 2>/dev/null; then
            cat > "${xml_file}" <<XMLEOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="${test_name}" tests="1" failures="1">
    <testcase name="PROCESS_CRASH" classname="${test_name}">
      <failure message="${test_name} crashed with $(signal_name ${rc}) (exit code ${rc}). Check coredumps in test-results/coredumps/.">
Process terminated by $(signal_name ${rc}).
Exit code: ${rc}
This may indicate a segfault, double-free, or stack overflow.
      </failure>
    </testcase>
  </testsuite>
</testsuites>
XMLEOF
        fi
        OVERALL_RC=1
        return 1
    fi

    # Extract which test cases failed
    local failed_tests
    failed_tests=$(extract_failed_tests "${xml_file}")

    if [ -z "${failed_tests}" ]; then
        echo "FAILED: ${test_name} failed but could not identify failing test cases"
        OVERALL_RC=1
        return 1
    fi

    local num_failed
    num_failed=$(echo "${failed_tests}" | wc -l)
    echo "INFO: ${num_failed} test case(s) failed in ${test_name}, retrying individually"

    # Retry each failing test case individually
    local all_passed=true
    while IFS= read -r tc; do
        local tc_passed=false
        for attempt in $(seq 1 "${GTEST_MAX_RETRIES}"); do
            local retry_xml="${RAPIDS_TESTS_DIR}/${test_name}-retry${attempt}-$(echo "${tc}" | tr '/' '_').xml"
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
                collect_coredumps "${test_name}-${tc}"
                break
            fi
        done

        if [ "${tc_passed}" = false ]; then
            echo "  FAILED: ${tc} failed after $((GTEST_MAX_RETRIES + 1)) attempts"
            all_passed=false
        fi
    done <<< "${failed_tests}"

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

# Report coredump summary
if ls "${COREDUMP_DIR}"/* &>/dev/null; then
    echo ""
    echo "=== COREDUMP SUMMARY ==="
    echo "Core dumps collected in ${COREDUMP_DIR}/:"
    ls -lh "${COREDUMP_DIR}/"
    echo "========================"
fi

exit ${OVERALL_RC}
