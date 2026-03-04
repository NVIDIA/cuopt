#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cuopt/cuopt as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cuopt_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../python/cuopt/cuopt/"

# Build the list of test directories based on CUOPT_TEST_COMPONENTS
COMPONENTS="${CUOPT_TEST_COMPONENTS:-all}"
TEST_DIRS=""

if [[ "${COMPONENTS}" == "all" ]]; then
    TEST_DIRS="tests"
else
    if [[ "${COMPONENTS}" == *"routing"* ]]; then
        TEST_DIRS="${TEST_DIRS} tests/routing"
    fi
    if [[ "${COMPONENTS}" == *"lp"* ]]; then
        TEST_DIRS="${TEST_DIRS} tests/linear_programming tests/quadratic_programming"
    fi
    # MIP does not have separate Python tests (tested through LP tests)

    # If no Python test dirs matched, skip
    if [[ -z "${TEST_DIRS}" ]]; then
        echo "No Python test directories match CUOPT_TEST_COMPONENTS=${COMPONENTS}, skipping."
        exit 0
    fi
fi

echo "Running pytest on: ${TEST_DIRS} (CUOPT_TEST_COMPONENTS=${COMPONENTS})"
# shellcheck disable=SC2086
pytest -s --cache-clear "$@" ${TEST_DIRS}
