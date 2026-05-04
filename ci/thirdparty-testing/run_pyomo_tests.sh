#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

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
timeout 5m python -m pytest \
    --verbose \
    --capture=no \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-thirdparty-pyomo.xml" \
    -k "cuopt or CUOPT" \
    pyomo/solvers/tests/

popd || exit 1
