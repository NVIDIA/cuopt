#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

# shellcheck source=ci/utils/crash_helpers.sh
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../utils/crash_helpers.sh"

rapids-logger "building 'pulp' from source and running cuOpt tests"

if [ -z "${PIP_CONSTRAINT:-}" ]; then
    rapids-logger "PIP_CONSTRAINT is not set; ensure ci/test_wheel_cuopt.sh (or equivalent) has set it so cuopt wheels are used."
    exit 1
fi

git clone --depth 1 https://github.com/coin-or/pulp.git
pushd ./pulp || exit 1

# Install PuLP in editable form so it uses the environment's cuopt (from PIP_CONSTRAINT)
python -m pip install \
    --constraint "${PIP_CONSTRAINT}" \
    --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    pytest \
    -e .

pip check

RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${PWD}/test-results}"
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-logger "running PuLP tests (cuOpt-related)"
# PuLP uses pytest; run only tests that reference cuopt/CUOPT
# Exit code 5 = no tests collected; then try run_tests.py which detects solvers (including cuopt)
pytest_rc=0
# test_numpy_float calls model.solve() with no explicit solver; PuLP's
# default-solver auto-detection list doesn't include CUOPT, so it raises
# "No solver available" in our cuopt-only test environment. Skip it here.
timeout 5m python -m pytest \
    --verbose \
    --capture=no \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-thirdparty-pulp.xml" \
    -k "cuopt or CUOPT" \
    --deselect pulp/tests/test_pulp.py::CUOPTTest::test_numpy_float \
    pulp/tests/ || pytest_rc=$?

if [ "$pytest_rc" -eq 5 ]; then
    rapids-logger "No pytest -k cuopt tests found; running PuLP run_tests.py (solver auto-detection, includes cuopt)"
    timeout 5m python pulp/tests/run_tests.py
    pytest_rc=$?
fi

# pytest's normal exit codes are 0-5 (passed / failed / interrupted /
# internal error / usage / no tests collected). Anything beyond that
# (timeout=124, signal deaths >128, etc.) means pytest did not finalize
# its JUnit XML, so synthesize a crash marker — otherwise nightly_report.py
# would see no failure and report "All tests passed."
if [ "${pytest_rc}" -gt 5 ]; then
    write_pytest_crash_marker "${RAPIDS_TESTS_DIR}/junit-thirdparty-pulp.xml" "thirdparty-pulp" "${pytest_rc}"
fi

popd || exit 1
exit "$pytest_rc"
