#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

# shellcheck source=ci/utils/crash_helpers.sh
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../utils/crash_helpers.sh"

rapids-logger "building 'pyomo' from source and running cuOpt tests"

if [ -z "${PIP_CONSTRAINT:-}" ]; then
    rapids-logger "PIP_CONSTRAINT is not set; ensure ci/test_wheel_cuopt.sh (or equivalent) has set it so cuopt wheels are used."
    exit 1
fi

git clone --depth 1 https://github.com/Pyomo/pyomo.git
pushd ./pyomo || exit 1

# Install Pyomo in editable form so it uses the environment's cuopt (from PIP_CONSTRAINT)
python -m pip install \
    --constraint "${PIP_CONSTRAINT}" \
    --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    pytest \
    -e .

pip check

RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${PWD}/test-results}"
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-logger "running Pyomo tests (cuopt_direct / cuOpt-related)"
# Run only tests that reference cuopt (cuopt_direct solver)
pytest_rc=0
timeout 5m python -m pytest \
    --verbose \
    --capture=no \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-thirdparty-pyomo.xml" \
    -k "cuopt or CUOPT" \
    pyomo/solvers/tests/ || pytest_rc=$?

# pytest's normal exit codes are 0-5 (passed / failed / interrupted /
# internal error / usage / no tests collected). Anything beyond that
# (timeout=124, signal deaths >128, etc.) means pytest did not finalize
# its JUnit XML, so synthesize a crash marker — otherwise nightly_report.py
# would see no failure and report "All tests passed."
if [ "${pytest_rc}" -gt 5 ]; then
    write_pytest_crash_marker "${RAPIDS_TESTS_DIR}/junit-thirdparty-pyomo.xml" "thirdparty-pyomo" "${pytest_rc}"
fi

popd || exit 1
exit "${pytest_rc}"
