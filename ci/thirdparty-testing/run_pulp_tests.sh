#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

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

rapids-logger "running PuLP tests (cuOpt-related)"
# PuLP uses pytest; run only tests that reference cuopt/CUOPT
# Exit code 5 = no tests collected; then try run_tests.py which detects solvers (including cuopt)
pytest_rc=0
timeout 5m python -m pytest \
    --verbose \
    --capture=no \
    -k "cuopt or CUOPT" \
    pulp/tests/ || pytest_rc=$?

if [ "$pytest_rc" -eq 5 ]; then
    rapids-logger "No pytest -k cuopt tests found; running PuLP run_tests.py (solver auto-detection, includes cuopt)"
    timeout 5m python pulp/tests/run_tests.py
    pytest_rc=$?
fi

popd || exit 1
exit "$pytest_rc"
