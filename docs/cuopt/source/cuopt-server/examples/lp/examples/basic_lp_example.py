# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Basic LP Server Example (Normal and Batch Mode)

This example demonstrates how to use the cuOpt Self-Hosted Service Client
to solve LP problems via the cuOpt server in both normal and batch modes.

Requirements:
    - cuOpt server running (default: localhost:5000)
    - cuopt_sh_client package installed

Problem:
    Minimize: -0.2*x1 + 0.1*x2
    Subject to:
        3.0*x1 + 4.0*x2 <= 5.4
        2.7*x1 + 10.1*x2 <= 4.9
        x1, x2 >= 0

The data is structured according to the OpenAPI specification (LPData):
    - csr_constraint_matrix: Constraint matrix in CSR format
    - constraint_bounds: Upper and lower bounds for constraints
    - objective_data: Objective coefficients and settings
    - variable_bounds: Variable bounds
    - solver_config: Solver settings

Expected Response:
    Normal mode: Single solution
    Batch mode: Array of solutions (one per problem)
"""

from cuopt_sh_client import CuOptServiceSelfHostClient
import json
import time


def repoll(cuopt_service_client, solution, repoll_tries):
    """
    Repoll the server for solution if it's still processing.

    If solver is still busy solving, the job will be assigned a request id
    and response is sent back in the format {"reqId": <REQUEST-ID>}.
    """
    if "reqId" in solution and "response" not in solution:
        req_id = solution["reqId"]
        for i in range(repoll_tries):
            solution = cuopt_service_client.repoll(
                req_id, response_type="dict"
            )
            if "reqId" in solution and "response" in solution:
                break

            # Sleep for a second before requesting
            time.sleep(1)

    return solution


def main():
    """Run the basic LP example in normal and batch modes."""
    # Example data for LP problem
    # The data is structured as per the OpenAPI specification for the server
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
        "solver_config": {"tolerances": {"optimality": 0.0001}},
    }

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost", port=5000, polling_timeout=25, timeout_exception=False
    )

    # Number of repoll requests to be carried out for a successful response
    repoll_tries = 500

    # Logging callback
    def log_callback(log):
        for i in log:
            print("server-log: ", i)

    print("=== Solving in Normal Mode ===")
    solution = cuopt_service_client.get_LP_solve(
        data, response_type="dict", logging_callback=log_callback
    )

    solution = repoll(cuopt_service_client, solution, repoll_tries)

    print("---------- Normal mode ---------------")
    print(json.dumps(solution, indent=4))

    print("\n=== Solving in Batch Mode ===")
    # For batch mode send list of mps/dict/DataModel
    solution = cuopt_service_client.get_LP_solve(
        [data, data], response_type="dict", logging_callback=log_callback
    )
    solution = repoll(cuopt_service_client, solution, repoll_tries)

    print("---------- Batch mode -----------------")
    print(json.dumps(solution, indent=4))


if __name__ == "__main__":
    main()
