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
if [ "${IS_NIGHTLY}" = "nightly" ]; then
    pytest -s --cache-clear --reruns 2 --reruns-delay 5 -p cuopt_rerun_xml "$@" tests || rc=$?
else
    pytest -s --cache-clear "$@" tests || rc=$?
fi

if [ "${rc}" -le 128 ]; then
    exit ${rc}
fi

echo "CRASH: pytest process died from $(signal_name ${rc}) (exit code ${rc})"

if [ "${IS_NIGHTLY}" != "nightly" ]; then
    exit ${rc}
fi

pytest_crash_isolate "${rc}" "${xml_file}"

exit ${rc}
