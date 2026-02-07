# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LP Warmstart Server Example

This example demonstrates how to use warmstart functionality with the cuOpt server.
Warmstart allows reusing solution context from a previous solve to speed up
solving of similar problems.

Note:
    Warmstart is only applicable to LP, not for MILP.

Requirements:
    - cuOpt server running (default: localhost:5000)
    - cuopt_sh_client package installed

Problem 1 & 2:
    Minimize: -0.2*x1 + 0.1*x2
    Subject to:
        3.0*x1 + 4.0*x2 <= 5.4
        2.7*x1 + 10.1*x2 <= 4.9
        x1, x2 >= 0

The second solve reuses the solution context from the first solve.
"""

from cuopt_sh_client import CuOptServiceSelfHostClient
import json


def main():
    """Run the warmstart LP example."""
    data = {
        "csr_constraint_matrix": {
            "offsets": [0, 2, 4],
            "indices": [0, 1, 0, 1],
            "values": [3.0, 4.0, 2.7, 10.1],
        },
        "constraint_bounds": {
            "upper_bounds": [5.4, 4.9],
            "lower_bounds": ["ninf", "ninf"],
        },
        "objective_data": {
            "coefficients": [-0.2, 0.1],
            "scalability_factor": 1.0,
            "offset": 0.0,
        },
        "variable_bounds": {
            "upper_bounds": ["inf", "inf"],
            "lower_bounds": [0.0, 0.0],
        },
        "maximize": False,
        "solver_config": {
            "tolerances": {"optimality": 0.0001},
            "pdlp_solver_mode": 1,  # Stable2
            "presolve": 0,
        },
    }

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost", port=5000, timeout_exception=False
    )

    print("=== Solving Problem 1 ===")
    # Set delete_solution to false so it can be used in next request
    initial_solution = cuopt_service_client.get_LP_solve(
        data, delete_solution=False, response_type="dict"
    )

    print(f"Problem 1 reqId: {initial_solution['reqId']}")
    print(
        f"Objective: {initial_solution['response']['solver_response']['solution']['primal_objective']}"
    )

    print("\n=== Solving Problem 2 with Warmstart ===")
    # Use previous solution saved in server as warmstart for this request.
    # That solution is referenced with previous request id.
    solution = cuopt_service_client.get_LP_solve(
        data, warmstart_id=initial_solution["reqId"], response_type="dict"
    )

    print(json.dumps(solution, indent=4))

    # Delete saved solution if not required to save space
    print("\n=== Cleaning Up ===")
    cuopt_service_client.delete(initial_solution["reqId"])
    print("Saved solution deleted")


if __name__ == "__main__":
    main()
