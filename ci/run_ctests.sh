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

# Routing test binaries to skip when SKIP_ROUTING_TESTS is set
ROUTING_TESTS=(
    ROUTING_TEST
    ROUTING_GES_TEST
    VEHICLE_ORDER_TEST
    VEHICLE_TYPES_TEST
    OBJECTIVE_FUNCTION_TEST
    RETAIL_L1TEST
    ROUTING_L1TEST
    ROUTING_UNIT_TEST
    WAYPOINT_MATRIXTEST
)

for gt in "${GTEST_DIR}"/*_TEST; do
    test_name=$(basename "${gt}")
    if [[ "${SKIP_ROUTING_TESTS:-}" == "true" ]]; then
        for routing_test in "${ROUTING_TESTS[@]}"; do
            if [[ "${test_name}" == "${routing_test}" ]]; then
                echo "Skipping routing gtest ${test_name} (SKIP_ROUTING_TESTS=true)"
                continue 2
            fi
        done
    fi
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
