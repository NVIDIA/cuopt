# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MILP Incumbent and Logging Callback Example

This example demonstrates how to use callbacks with the cuOpt server:
- Incumbent solution callback: Receives intermediate solutions as they're found
- Logging callback: Receives solver log messages in real-time

Note:
    Incumbent solution callback is only applicable to MILP, not for LP.

Requirements:
    - cuOpt server running (default: localhost:5000)
    - cuopt_sh_client package installed

Problem:
    Maximize: 1.2*x + 1.7*y
    Subject to:
        x + y <= 5000
        x, y are integers
        0 <= x <= 3000
        0 <= y <= 5000

Expected Output:
    server-log: Solving a problem with 1 constraints 2 variables (2 integers) and 2 nonzeros
    ...
    Solution : [0.0, 5000.0] cost : 8500.0
"""

from cuopt_sh_client import CuOptServiceSelfHostClient
import json


def main():
    """Run the incumbent and logging callback example."""
    data = {
        "csr_constraint_matrix": {
            "offsets": [0, 2],
            "indices": [0, 1],
            "values": [1.0, 1.0],
        },
        "constraint_bounds": {"upper_bounds": [5000.0], "lower_bounds": [0.0]},
        "objective_data": {
            "coefficients": [1.2, 1.7],
            "scalability_factor": 1.0,
            "offset": 0.0,
        },
        "variable_bounds": {
            "upper_bounds": [3000.0, 5000.0],
            "lower_bounds": [0.0, 0.0],
        },
        "maximize": True,
        "variable_names": ["x", "y"],
        "variable_types": ["I", "I"],
        "solver_config": {"time_limit": 30},
    }

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost", port=5000, timeout_exception=False
    )

    # Incumbent callback - receives intermediate host solutions
    def callback(solution, solution_cost, solution_bound):
        """Called when solver finds a new incumbent solution."""
        print(
            f"Solution : {solution} cost : {solution_cost} "
            f"bound : {solution_bound}\n"
        )

    # Logging callback - receives server log messages
    def log_callback(log):
        """Called when server sends log messages."""
        for i in log:
            print("server-log: ", i)

    print("=== Solving MILP with Callbacks ===")
    print("\n--- Logging Output ---")

    solution = cuopt_service_client.get_LP_solve(
        data,
        incumbent_callback=callback,
        response_type="dict",
        logging_callback=log_callback,
    )

    print("\n--- Final Solution ---")
    print(json.dumps(solution, indent=4))


if __name__ == "__main__":
    main()
