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

# Write a JUnit XML crash marker to a file.
# This records a crash as a test failure so nightly_report.py can track it.
#
# Usage: write_crash_xml <xml_file> <suite_name> <test_name> <message> <detail>
write_crash_xml() {
    local xml_file="$1"
    local suite_name="$2"
    local test_name="$3"
    local message="$4"
    local detail="$5"

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
