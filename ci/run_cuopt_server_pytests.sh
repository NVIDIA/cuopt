#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cuopt_server/cuopt_server as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cuopt_server_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuopt_server/cuopt_server/

PYTEST_ARGS=("$@")
if [[ "${SKIP_ROUTING_TESTS:-}" == "true" ]]; then
    echo "Skipping routing tests (SKIP_ROUTING_TESTS=true)"
    PYTEST_ARGS+=(
        "--ignore=tests/test_server.py"
        "--ignore=tests/test_set_cost_matrix.py"
        "--ignore=tests/test_set_cost_waypoint_graph.py"
        "--ignore=tests/test_set_fleet_data.py"
        "--ignore=tests/test_set_task_data.py"
        "--ignore=tests/test_set_travel_time_waypoint_graph.py"
        "--ignore=tests/test_initial_solutions.py"
        "--ignore=tests/test_multi_cost.py"
    )
fi

pytest -s --cache-clear "${PYTEST_ARGS[@]}" tests
