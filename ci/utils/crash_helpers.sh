#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared helpers for crash detection and JUnit XML crash markers.
# Source this from test runner scripts (run_ctests.sh, run_cuopt_pytests.sh, etc.)

# Convert exit code > 128 to a human-readable signal name.
signal_name() {
    local sig=$(($1 - 128))
    case "${sig}" in
        6)  echo "SIGABRT" ;;
        11) echo "SIGSEGV (segfault)" ;;
        *)  echo "signal ${sig}" ;;
    esac
}

# Check if an exit code indicates signal death (exit code > 128).
was_signal_death() {
    [ "$1" -gt 128 ]
}

# Escape XML special characters in a string.
# Replaces &, <, >, and " with their XML entity equivalents.
xml_escape() {
    local s="$1"
    s=$(printf '%s' "$s" | sed -e 's/&/\&amp;/g' \
                                -e 's/</\&lt;/g' \
                                -e 's/>/\&gt;/g' \
                                -e 's/"/\&quot;/g')
    printf '%s' "$s"
}

# Write a JUnit XML crash marker to a file.
# This records a crash as a test failure so nightly_report.py can track it.
#
# Usage: write_crash_xml <xml_file> <suite_name> <test_name> <message> <detail>
write_crash_xml() {
    local xml_file="$1"
    local suite_name
    local test_name
    local message
    local detail
    suite_name=$(xml_escape "$2")
    test_name=$(xml_escape "$3")
    message=$(xml_escape "$4")
    detail=$(xml_escape "$5")

    cat > "${xml_file}" <<XMLEOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="${suite_name}" tests="1" failures="1">
    <testcase name="${test_name}" classname="${suite_name}">
      <failure message="${message}">
${detail}
      </failure>
    </testcase>
  </testsuite>
</testsuites>
XMLEOF
}

# Isolate crashing pytest tests by retrying individually.
# Called after pytest exits with a signal (exit code > 128) on nightly builds.
#
# Requires: RAPIDS_TESTS_DIR, PYTEST_MAX_CRASH_RETRIES, SCRIPT_DIR (for junit_helpers.py)
# Usage: pytest_crash_isolate <exit_code> <xml_file>
pytest_crash_isolate() {
    local rc="$1"
    local xml_file="$2"

    echo "INFO: Collecting test list for individual retry..."
    local test_list
    test_list=$(pytest --collect-only -q tests 2>/dev/null | grep "::" | head -500 || echo "")

    if [ -z "${test_list}" ]; then
        echo "FAILED: Could not collect test list, cannot isolate crashing test"
        if [ -n "${xml_file}" ]; then
            # Write crash marker to a separate file to preserve any partial
            # results already written to xml_file by the crashed pytest run
            local crash_marker="${RAPIDS_TESTS_DIR}/crash-marker-collection-failed.xml"
            write_crash_xml "${crash_marker}" "pytest-crash" "PROCESS_CRASH" \
                "pytest crashed with $(signal_name "${rc}") (exit code ${rc})" \
                "pytest process terminated by $(signal_name "${rc}"). Could not collect test list for retry."
        fi
        return
    fi

    # Extract tests that already passed from partial JUnit XML (if any)
    local passed_tests=""
    if [ -n "${xml_file}" ] && [ -f "${xml_file}" ]; then
        passed_tests=$(python3 "${SCRIPT_DIR}/utils/junit_helpers.py" passed "${xml_file}" --sep "::" 2>/dev/null || echo "")
    fi

    # Only retry tests that didn't already pass
    if [ -n "${passed_tests}" ]; then
        local num_passed
        num_passed=$(echo "${passed_tests}" | wc -l)
        echo "INFO: ${num_passed} tests already passed before crash, skipping those"
        test_list=$(comm -23 \
            <(echo "${test_list}" | sort) \
            <(echo "${passed_tests}" | sort))
    fi

    local num_tests
    num_tests=$(echo "${test_list}" | grep -c '.' || echo "0")
    if [ "${num_tests}" -eq 0 ]; then
        echo "INFO: All tests already passed before crash, nothing to retry"
        return
    fi
    echo "INFO: Retrying ${num_tests} tests individually to isolate crash"

    local crash_tests=()
    local flaky_crash_tests=()

    while IFS= read -r test_id; do
        [ -z "${test_id}" ] && continue
        local safe_name
        safe_name=$(echo "${test_id}" | tr -c '[:alnum:]._-' '_')

        for attempt in $(seq 1 "${PYTEST_MAX_CRASH_RETRIES}"); do
            local retry_rc=0
            local retry_xml="${RAPIDS_TESTS_DIR}/crash-retry${attempt}-${safe_name}.xml"
            pytest -s --no-header -x --junitxml="${retry_xml}" "${test_id}" 2>/dev/null || retry_rc=$?

            if [ "${retry_rc}" -eq 0 ]; then
                if [ "${attempt}" -gt 1 ]; then
                    echo "  FLAKY-CRASH: ${test_id} — crashed then passed on retry ${attempt}"
                    flaky_crash_tests+=("${test_id}")
                fi
                break
            elif [ "${retry_rc}" -gt 128 ]; then
                echo "  CRASH: ${test_id} — $(signal_name "${retry_rc}") on attempt ${attempt}"
                if [ "${attempt}" -eq "${PYTEST_MAX_CRASH_RETRIES}" ]; then
                    echo "  FAILED: ${test_id} — crashes consistently"
                    crash_tests+=("${test_id}")
                    write_crash_xml "${retry_xml}" "pytest-crash" "${test_id}" \
                        "${test_id} crashed with $(signal_name "${retry_rc}") on ${attempt} attempts" \
                        "Consistent crash: $(signal_name "${retry_rc}"). This test needs urgent investigation."
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
}
