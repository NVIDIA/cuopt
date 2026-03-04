#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cuopt/cuopt as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cuopt_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuopt/cuopt/

# Skip routing tests when CUOPT_RUN_ROUTING_TESTS is false (e.g. PRs that don't touch routing)
if [[ "${CUOPT_RUN_ROUTING_TESTS:-true}" == "false" ]]; then
  pytest -s --cache-clear "$@" --ignore=tests/routing tests
else
  pytest -s --cache-clear "$@" tests
fi
