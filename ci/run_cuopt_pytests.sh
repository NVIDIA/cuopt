#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cuopt/cuopt as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Resolve paths before cd (BASH_SOURCE is relative and won't resolve after cd)
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# shellcheck source=ci/utils/crash_helpers.sh
source "${SCRIPT_DIR}/utils/crash_helpers.sh"

# Support invoking run_cuopt_pytests.sh outside the script directory
cd "${SCRIPT_DIR}/../python/cuopt/cuopt/"

RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${PWD}/test-results}"
export RAPIDS_TESTS_DIR
PYTEST_MAX_CRASH_RETRIES=${PYTEST_MAX_CRASH_RETRIES:-2}
IS_NIGHTLY="${RAPIDS_BUILD_TYPE:-}"

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
# A test that never returns takes the whole step down when the outer 'timeout'
# fires, and pytest is killed before it can say which test was running. -v names
# each test as it is dispatched, and faulthandler_timeout dumps the stack of any
# test still running after FAULTHANDLER_TIMEOUT seconds, so a stuck test
# identifies itself while the step is still alive. Neither kills the test; they
# only make it visible.
FAULTHANDLER_TIMEOUT=${FAULTHANDLER_TIMEOUT:-600}
PYTEST_DIAG_ARGS=(-v -o "faulthandler_timeout=${FAULTHANDLER_TIMEOUT}" --durations=25)

if [ "${IS_NIGHTLY}" = "nightly" ]; then
    pytest -s --cache-clear --reruns 2 --reruns-delay 5 -p cuopt_rerun_xml \
        "${PYTEST_DIAG_ARGS[@]}" "$@" tests || rc=$?
else
    # loadgroup keeps xdist_group (grpc server) tests on one worker;
    # max-worker-restart=0 stops a crashed worker from respawning.
    pytest -s --cache-clear -n 4 --dist loadgroup --max-worker-restart=0 \
        "${PYTEST_DIAG_ARGS[@]}" "$@" tests || rc=$?
fi

# If not a crash, exit normally
if [ "${rc}" -le 128 ]; then
    exit ${rc}
fi

echo "CRASH: pytest process died from $(signal_name ${rc}) (exit code ${rc})"

# For non-nightly builds, fail immediately — no crash isolation. But
# still write a synthetic crash XML so nightly_report.py reports the
# failure (pytest didn't finalize JUnit on a mid-run crash).
if [ "${IS_NIGHTLY}" != "nightly" ]; then
    write_pytest_crash_marker "${xml_file}" "pytest-cuopt" "${rc}"
    exit ${rc}
fi

pytest_crash_isolate "${rc}" "${xml_file}"

exit ${rc}
