#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run cuOpt C++ tests through CTest.
#
# CTest is required because the CMake test registry carries per-test metadata
# that direct gtest execution loses. In particular, CLI_TEST sets PATH so the
# test shells out to the freshly built cuopt_cli rather than an installed or
# stale copy.
#
# Environment variables:
#   CTEST_DIR        - CMake build tree containing CTestTestfile.cmake
#                     (default: cpp/build/latest)
#   RAPIDS_TESTS_DIR - directory for test results

set -euo pipefail

script_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
repo_dir="$(realpath "${script_dir}/..")"

export RAPIDS_DATASET_ROOT_DIR="${RAPIDS_DATASET_ROOT_DIR:-${repo_dir}/datasets}"
export CUOPT_HOME="${CUOPT_HOME:-${repo_dir}}"

build_ctest_dir="${repo_dir}/cpp/build/latest"
installed_ctest_dir="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcuopt"

if [[ -n "${CTEST_DIR:-}" ]]; then
    ctest_dir="${CTEST_DIR}"
elif [[ -f "${build_ctest_dir}/CTestTestfile.cmake" ]]; then
    ctest_dir="${build_ctest_dir}"
elif [[ -f "${installed_ctest_dir}/CTestTestfile.cmake" ]]; then
    ctest_dir="${installed_ctest_dir}"
else
    echo "Error: CTest registry not found. Searched:" >&2
    echo "  - ${build_ctest_dir}/CTestTestfile.cmake" >&2
    echo "  - ${installed_ctest_dir}/CTestTestfile.cmake" >&2
    exit 1
fi

RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR:-${repo_dir}/test-results}"
mkdir -p "${RAPIDS_TESTS_DIR}"
ctest_junit="${RAPIDS_TESTS_DIR}/ctest.xml"

echo "Running ctest from ${ctest_dir}"
ctest --test-dir "${ctest_dir}" --output-on-failure --output-junit "${ctest_junit}" "$@"
