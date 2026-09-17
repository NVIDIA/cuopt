#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

# shellcheck source=ci/utils/crash_helpers.sh
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../utils/crash_helpers.sh"

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

RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${PWD}/test-results}"
mkdir -p "${RAPIDS_TESTS_DIR}"
# Canonicalize to an absolute path: if a caller exports RAPIDS_TESTS_DIR as a
# relative path, it would resolve differently after the popd below (relative
# to the clone we're leaving vs. relative to where we land), splitting the
# junit output across two directories.
RAPIDS_TESTS_DIR="$(cd -- "${RAPIDS_TESTS_DIR}" && pwd -P)"

# Leave the clone: cwd is 'cvxpy/' containing a 'cvxpy/' package
# subdirectory, and Python puts cwd first on sys.path, so importing 'cvxpy'
# from here silently shadows the installed wheel with the uncompiled source
# tree -- producing "ImportError: cannot import name '_cvxcore'" even on a
# perfectly good build/install. This is the actual root cause of the
# nightly failure fixed here.
popd

echo "running 'cvxpy' tests"
pytest_rc=0
# --pyargs (module path, not a filesystem path) avoids pytest re-inserting
# the clone root onto sys.path via its rootdir walk-up, which would
# reintroduce the shadowing above even with cwd fixed.
timeout 3m python -m pytest \
    --verbose \
    --capture=no \
    --error-for-skips \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-thirdparty-cvxpy.xml" \
    -k "TestCUOPT" \
    --pyargs cvxpy.tests.test_conic_solvers || pytest_rc=$?

# pytest's normal exit codes are 0-5 (passed / failed / interrupted /
# internal error / usage / no tests collected). Anything beyond that
# (timeout=124, signal deaths >128, etc.) means pytest did not finalize
# its JUnit XML, so synthesize a crash marker — otherwise nightly_report.py
# would see no failure and report "All tests passed."
if [ "${pytest_rc}" -gt 5 ]; then
    write_pytest_crash_marker "${RAPIDS_TESTS_DIR}/junit-thirdparty-cvxpy.xml" "thirdparty-cvxpy" "${pytest_rc}"
fi

exit "${pytest_rc}"
