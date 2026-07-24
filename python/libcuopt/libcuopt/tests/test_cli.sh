#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Add cuopt_cli path to PATH variable
if command -v pyenv &> /dev/null; then
    PATH="$(pyenv root)/versions/$(pyenv version-name)/bin:$PATH"
    export PATH
fi

# Test the CLI

# Add a test for the help command
cuopt_cli --help | grep "Usage: cuopt_cli" > /dev/null || (echo "Expected usage information not found" && exit 1)

# Add a test with a simple linear programming problem

# Run solver and check for optimal status - fail if not found

cuopt_cli "${RAPIDS_DATASET_ROOT_DIR}"/linear_programming/good-mps-1.mps | grep -q "Status: " || (echo "Expected solution not found" && exit 1)

cuopt_cli "${RAPIDS_DATASET_ROOT_DIR}"/linear_programming/good-mps-1.lp | grep -q "Status: " || (echo "Expected solution not found for .lp" && exit 1)

cuopt_cli "${RAPIDS_DATASET_ROOT_DIR}"/linear_programming/good-mps-1.lp.gz | grep -q "Status: " || (echo "Expected solution not found for .lp.gz" && exit 1)

cuopt_cli "${RAPIDS_DATASET_ROOT_DIR}"/linear_programming/good-mps-1.lp.bz2 | grep -q "Status: " || (echo "Expected solution not found for .lp.bz2" && exit 1)

# Add a for mixed integer programming test with options

cuopt_cli "${RAPIDS_DATASET_ROOT_DIR}"/mip/sample.mps --mip-absolute-gap 0.01 --time-limit 10 | grep -q "Best objective" || (echo "Expected solution objective not found" && exit 1)
