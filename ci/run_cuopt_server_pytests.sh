#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cuopt_server/cuopt_server as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Resolve paths before cd (BASH_SOURCE is relative and won't resolve after cd)
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# shellcheck source=ci/utils/crash_helpers.sh
source "${SCRIPT_DIR}/utils/crash_helpers.sh"

# Support invoking run_cuopt_server_pytests.sh outside the script directory
cd "${SCRIPT_DIR}/../python/cuopt_server/cuopt_server/"

RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${PWD}/test-results}"
export RAPIDS_TESTS_DIR
PYTEST_MAX_CRASH_RETRIES=${PYTEST_MAX_CRASH_RETRIES:-2}
IS_NIGHTLY="${RAPIDS_BUILD_TYPE:-}"

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

if [ "${rc}" -le 128 ]; then
    exit ${rc}
fi

echo "CRASH: pytest process died from $(signal_name ${rc}) (exit code ${rc})"

if [ "${IS_NIGHTLY}" != "nightly" ]; then
    exit ${rc}
fi

echo "INFO: Collecting test list for individual retry..."
test_list=$(pytest --collect-only -q tests 2>/dev/null | grep "::" | head -500 || echo "")

if [ -z "${test_list}" ]; then
    echo "FAILED: Could not collect test list, cannot isolate crashing test"
    if [ -n "${xml_file}" ]; then
        write_crash_xml "${xml_file}" "pytest-crash" "PROCESS_CRASH" \
            "pytest crashed with $(signal_name ${rc}) (exit code ${rc})" \
            "pytest process terminated by $(signal_name ${rc}). Could not collect test list for retry."
    fi
    exit ${rc}
fi

passed_tests=""
if [ -n "${xml_file}" ] && [ -f "${xml_file}" ]; then
    passed_tests=$(python3 "${SCRIPT_DIR}/utils/junit_helpers.py" passed "${xml_file}" --sep "::" 2>/dev/null || echo "")
fi

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
                write_crash_xml "${retry_xml}" "pytest-crash" "${test_id}" \
                    "${test_id} crashed with $(signal_name ${retry_rc}) on ${attempt} attempts" \
                    "Consistent crash: $(signal_name ${retry_rc}). This test needs urgent investigation."
            fi
        else
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
