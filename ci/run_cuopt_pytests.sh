#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cuopt/cuopt as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cuopt_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuopt/cuopt/

PYTEST_ARGS=("$@")
if [[ "${SKIP_ROUTING_TESTS:-}" == "true" ]]; then
    echo "Skipping routing tests (SKIP_ROUTING_TESTS=true)"
    PYTEST_ARGS+=("--ignore=tests/routing")
fi

pytest -s --cache-clear "${PYTEST_ARGS[@]}" tests
