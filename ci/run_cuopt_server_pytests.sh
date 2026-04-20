#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cuopt_server/cuopt_server as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cuopt_server_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuopt_server/cuopt_server/

RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${PWD}/test-results}"

rc=0
pytest -s --cache-clear --reruns 2 --reruns-delay 5 "$@" tests || rc=$?

# Detect signal death (segfault, abort)
if [ "${rc}" -gt 128 ]; then
    sig=$((rc - 128))
    case "${sig}" in
        6)  signame="SIGABRT" ;;
        11) signame="SIGSEGV (segfault)" ;;
        *)  signame="signal ${sig}" ;;
    esac
    echo "CRASH: pytest process died from ${signame} (exit code ${rc})"

    for arg in "$@"; do
        if [[ "${arg}" == *"junitxml"* ]]; then
            xml_file="${arg#*=}"
            if [ ! -f "${xml_file}" ] || ! grep -q 'testcase' "${xml_file}" 2>/dev/null; then
                cat > "${xml_file}" <<XMLEOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="pytest-crash" tests="1" failures="1">
    <testcase name="PROCESS_CRASH" classname="pytest">
      <failure message="pytest crashed with ${signame} (exit code ${rc})">
pytest process terminated by ${signame}.
Exit code: ${rc}
This may indicate a segfault in a C extension, CUDA driver issue, or memory corruption.
      </failure>
    </testcase>
  </testsuite>
</testsuites>
XMLEOF
                echo "CRASH: Wrote crash marker to ${xml_file}"
            fi
            break
        fi
    done
fi

exit ${rc}
