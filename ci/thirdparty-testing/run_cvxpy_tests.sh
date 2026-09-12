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

# cvxpy compiles its '_cvxcore' extension with g++ during 'pip wheel'; without
# it the build fails loudly (see #1747).
if ! command -v g++ >/dev/null 2>&1; then
    echo "g++ not found; attempting to install build-essential"
    if command -v apt-get >/dev/null 2>&1; then
        SUDO=""
        if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
            SUDO="sudo"
        fi
        ${SUDO} apt-get update -y && ${SUDO} apt-get install -y --no-install-recommends build-essential
    else
        echo "FATAL: no apt-get available to install a C++ compiler; cannot build cvxpy's _cvxcore extension"
        exit 1
    fi
fi

pip wheel \
    -w dist \
    .

# Fail loudly here if the wheel is missing the compiled extension, rather
# than downstream as a confusing test-time ImportError.
cvxpy_wheel="$(echo ./dist/cvxpy*.whl)"
if ! python -m zipfile -l "${cvxpy_wheel}" | grep -q 'cvxpy/cvxcore/python/_cvxcore.*\.so'; then
    echo "FATAL: built cvxpy wheel '${cvxpy_wheel}' does not contain a compiled"
    echo "       '_cvxcore' extension (.so). The build likely silently skipped"
    echo "       compiling it -- check for a missing/broken C++ toolchain or an"
    echo "       unexpected 'PYODIDE' env var. Wheel contents:"
    python -m zipfile -l "${cvxpy_wheel}"
    exit 1
fi

# NOTE: installing cvxpy[CUOPT] alongside CI artifacts is helpful to catch dependency conflicts
echo "installing 'cvxpy' with cuopt"
python -m pip install \
    --constraint "${PIP_CONSTRAINT}" \
    --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
    'pytest-error-for-skips>=2.0.2' \
    "${cvxpy_wheel}[CUOPT,testing]"

# ensure that environment is still consistent (i.e. cvxpy requirements do not conflict with cuopt's)
pip check

RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${PWD}/test-results}"
mkdir -p "${RAPIDS_TESTS_DIR}"

# Leave the clone: cwd is 'cvxpy/' containing a 'cvxpy/' package
# subdirectory, and Python puts cwd first on sys.path, so importing 'cvxpy'
# from here silently shadows the installed wheel with the uncompiled source
# tree -- producing "ImportError: cannot import name '_cvxcore'" even on a
# perfectly good build/install. This is the actual root cause of the
# nightly failure fixed here.
popd

# Belt-and-braces: verify the installed cvxpy can actually load its native
# backend before running any tests, so a broken extension fails here with a
# clear message instead of as a wall of downstream test ImportErrors.
cvxcore_import_check_log="$(mktemp)"
if ! python -c "from cvxpy.cvxcore.python.cppbackend import build_matrix" 2>"${cvxcore_import_check_log}"; then
    echo "FATAL: installed cvxpy cannot import its '_cvxcore' native backend:"
    cat "${cvxcore_import_check_log}"
    rm -f "${cvxcore_import_check_log}"
    exit 1
fi
rm -f "${cvxcore_import_check_log}"

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
