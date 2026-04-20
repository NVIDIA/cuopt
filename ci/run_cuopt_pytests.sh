#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cuopt/cuopt as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Resolve paths before cd (BASH_SOURCE is relative and won't resolve after cd)
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# Support invoking run_cuopt_pytests.sh outside the script directory
cd "${SCRIPT_DIR}/../python/cuopt/cuopt/"

RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${PWD}/test-results}"
export RAPIDS_TESTS_DIR
PYTEST_MAX_CRASH_RETRIES=${PYTEST_MAX_CRASH_RETRIES:-2}
IS_NIGHTLY="${RAPIDS_BUILD_TYPE:-}"

signal_name() {
    local sig=$(($1 - 128))
    case "${sig}" in
        6)  echo "SIGABRT" ;;
        11) echo "SIGSEGV (segfault)" ;;
        *)  echo "signal ${sig}" ;;
    esac
}

# Extract junitxml path from args
xml_file=""
for arg in "$@"; do
    if [[ "${arg}" == *"junitxml"* ]]; then
        xml_file="${arg#*=}"
        break
    fi
done

# Add CI utils to PYTHONPATH so the rerun XML plugin is importable
export PYTHONPATH="${SCRIPT_DIR}/utils:${PYTHONPATH:-}"

rc=0
pytest -s --cache-clear --reruns 2 --reruns-delay 5 -p cuopt_rerun_xml "$@" tests || rc=$?

# If not a crash, exit normally
if [ "${rc}" -le 128 ]; then
    exit ${rc}
fi

echo "CRASH: pytest process died from $(signal_name ${rc}) (exit code ${rc})"

# For non-nightly builds, fail immediately — no crash isolation
if [ "${IS_NIGHTLY}" != "nightly" ]; then
    exit ${rc}
fi

# Collect test list and retry individually to find the crashing test
echo "INFO: Collecting test list for individual retry..."
test_list=$(pytest --collect-only -q tests 2>/dev/null | grep "::" | head -500 || echo "")

if [ -z "${test_list}" ]; then
    echo "FAILED: Could not collect test list, cannot isolate crashing test"
    if [ -n "${xml_file}" ]; then
        cat > "${xml_file}" <<XMLEOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="pytest-crash" tests="1" failures="1">
    <testcase name="PROCESS_CRASH" classname="pytest">
      <failure message="pytest crashed with $(signal_name ${rc}) (exit code ${rc})">
pytest process terminated by $(signal_name ${rc}).
Could not collect test list for retry.
      </failure>
    </testcase>
  </testsuite>
</testsuites>
XMLEOF
    fi
    exit ${rc}
fi

# Extract tests that already passed from partial JUnit XML (if any)
passed_tests=""
if [ -n "${xml_file}" ] && [ -f "${xml_file}" ]; then
    passed_tests=$(python3 -c "
import sys
from xml.etree import ElementTree
try:
    tree = ElementTree.parse(sys.argv[1])
    for tc in tree.iter('testcase'):
        if tc.find('failure') is None and tc.find('error') is None and tc.find('skipped') is None:
            cls = tc.get('classname', '')
            name = tc.get('name', '')
            if cls and name:
                print(f'{cls}::{name}')
except Exception:
    pass
" "${xml_file}" 2>/dev/null || echo "")
fi

# Only retry tests that didn't already pass
if [ -n "${passed_tests}" ]; then
    num_passed=$(echo "${passed_tests}" | wc -l)
    echo "INFO: ${num_passed} tests already passed before crash, skipping those"
    test_list=$(comm -23 \
        <(echo "${test_list}" | sort) \
        <(echo "${passed_tests}" | sort))
fi

num_tests=$(echo "${test_list}" | grep -c '.' || echo "0")
if [ "${num_tests}" -eq 0 ]; then
    echo "INFO: All tests already passed before crash, nothing to retry"
    exit ${rc}
fi
echo "INFO: Retrying ${num_tests} tests individually to isolate crash"

crash_tests=()
flaky_crash_tests=()

while IFS= read -r test_id; do
    [ -z "${test_id}" ] && continue
    safe_name=$(echo "${test_id}" | tr '/:' '__')

    for attempt in $(seq 1 "${PYTEST_MAX_CRASH_RETRIES}"); do
        retry_rc=0
        retry_xml="${RAPIDS_TESTS_DIR}/crash-retry${attempt}-${safe_name}.xml"
        pytest -s --no-header -x --junitxml="${retry_xml}" "${test_id}" 2>/dev/null || retry_rc=$?

        if [ "${retry_rc}" -eq 0 ]; then
            if [ "${attempt}" -gt 1 ]; then
                echo "  FLAKY-CRASH: ${test_id} — crashed then passed on retry ${attempt}"
                flaky_crash_tests+=("${test_id}")
            fi
            break
        elif [ "${retry_rc}" -gt 128 ]; then
            echo "  CRASH: ${test_id} — $(signal_name ${retry_rc}) on attempt ${attempt}"
            if [ "${attempt}" -eq "${PYTEST_MAX_CRASH_RETRIES}" ]; then
                echo "  FAILED: ${test_id} — crashes consistently"
                crash_tests+=("${test_id}")
                cat > "${retry_xml}" <<XMLEOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="pytest-crash" tests="1" failures="1">
    <testcase name="${test_id}" classname="pytest-crash">
      <failure message="${test_id} crashed with $(signal_name ${retry_rc}) on ${attempt} attempts">
Consistent crash: $(signal_name ${retry_rc}).
This test needs urgent investigation.
      </failure>
    </testcase>
  </testsuite>
</testsuites>
XMLEOF
            fi
        else
            # Normal test failure, not a crash — already in retry_xml
            break
        fi
    done
done <<< "${test_list}"

echo ""
echo "=== CRASH ISOLATION SUMMARY ==="
echo "Consistent crashes: ${#crash_tests[@]}"
for t in "${crash_tests[@]+"${crash_tests[@]}"}"; do echo "  :x: ${t}"; done
echo "Flaky crashes (passed on retry): ${#flaky_crash_tests[@]}"
for t in "${flaky_crash_tests[@]+"${flaky_crash_tests[@]}"}"; do echo "  :warning: ${t}"; done
echo "================================"

exit ${rc}
