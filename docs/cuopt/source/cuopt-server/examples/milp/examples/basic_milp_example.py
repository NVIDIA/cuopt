# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Basic MILP Server Example

This example demonstrates how to use the cuOpt Self-Hosted Service Client
to solve MILP (Mixed Integer Linear Programming) problems via the cuOpt server.

The major difference between this example and the LP example is that some of
the variables are integers, so 'variable_types' need to be specified.

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

Expected Response:
    {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {
                    "primal_objective": 8500.0,
                    "vars": {
                        "x": 0.0,
                        "y": 5000.0
                    }
                }
            }
        }
    }
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
    """Run the basic MILP example."""
    # Example data for MILP problem
    # The data is structured as per the OpenAPI specification for the server
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
        "variable_types": ["I", "I"],  # Both variables are integers
        "solver_config": {"time_limit": 30},
    }

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost", port=5000, polling_timeout=25, timeout_exception=False
    )

    print("=== Solving MILP Problem ===")
    solution = cuopt_service_client.get_LP_solve(data, response_type="dict")

    # Number of repoll requests to be carried out for a successful response
    repoll_tries = 500

    solution = repoll(cuopt_service_client, solution, repoll_tries)

    print(json.dumps(solution, indent=4))


if __name__ == "__main__":
    main()
