#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support customizing the gtests' install location
# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcuopt/"
# Fall back to the build directory (devcontainer environments)
devcontainers_test_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest/gtests/libcuopt/"

if [[ -d "${installed_test_location}" ]]; then
    GTEST_DIR="${installed_test_location}"
elif [[ -d "${devcontainers_test_location}" ]]; then
    GTEST_DIR="${devcontainers_test_location}"
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_test_location}" >&2
    exit 1
fi

for gt in "${GTEST_DIR}"/*_TEST; do
    test_name=$(basename "${gt}")
    echo "Running gtest ${test_name}"
    "${gt}" "$@"
done

# Run C_API_TEST with CPU memory for local solves (excluding time limit tests)
if [ -x "${GTEST_DIR}/C_API_TEST" ]; then
  echo "Running gtest C_API_TEST with CUOPT_USE_CPU_MEM_FOR_LOCAL"
  CUOPT_USE_CPU_MEM_FOR_LOCAL=1 "${GTEST_DIR}/C_API_TEST" --gtest_filter=-c_api/TimeLimitTestFixture.* "$@"
else
  echo "Skipping C_API_TEST with CUOPT_USE_CPU_MEM_FOR_LOCAL (binary not found)"
fi
