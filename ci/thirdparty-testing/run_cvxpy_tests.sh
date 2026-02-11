#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

echo "building 'cvxpy' from source"

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
    echo "Skipping cvxpy tests: Python version is less than 3.11 (found $PYTHON_VERSION)"
    exit 0
fi

git clone https://github.com/cvxpy/cvxpy.git
pushd ./cvxpy || exit 1
pip wheel \
    -w dist \
    .

# NOTE: installing cvxpy[CUOPT] alongside CI artifacts is helpful to catch dependency conflicts
echo "installing 'cvxpy' with cuopt"
python -m pip install \
    --constraint "${PIP_CONSTRAINT}" \
    --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    'pytest-error-for-skips>=2.0.2' \
    "$(echo ./dist/cvxpy*.whl)[CUOPT,testing]"

# ensure that environment is still consistent (i.e. cvxpy requirements do not conflict with cuopt's)
pip check

echo "running 'cvxpy' tests"
timeout 3m python -m pytest \
    --verbose \
    --capture=no \
    --error-for-skips \
    -k "TestCUOPT" \
    ./cvxpy/tests/test_conic_solvers.py
