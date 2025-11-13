# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Basic Routing Server Example

This example demonstrates how to use the cuOpt Self-Hosted Service Client
to solve a simple routing problem via the cuOpt server.

Requirements:
    - cuOpt server running (default: localhost:5000)
    - cuopt_sh_client package installed

Problem:
    - 2 locations (0 and 1)
    - 2 tasks at locations 0 and 1
    - 2 vehicles starting and ending at location [0, 0]

The data is structured according to the OpenAPI specification:
    - cost_matrix_data: Cost matrix indexed by string keys
    - task_data: Task locations
    - fleet_data: Vehicle starting locations

Expected Response:
    {
        "response": {
            "solver_response": {
                "status": 0,
                "num_vehicles": 1,
                "solution_cost": 2.0,
                "vehicle_data": {
                    "0": {
                        "task_id": ["Depot", "0", "1", "Depot"],
                        "route": [0, 0, 1, 0]
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
    Solver needs to be re-polled for response using this <REQUEST-ID>.
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
    """Run the basic routing example with cuOpt server."""
    # Example data for routing problem
    # The data is structured as per the OpenAPI specification for the server
    data = {
        "cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
        "task_data": {"task_locations": [0, 1]},
        "fleet_data": {"vehicle_locations": [[0, 0], [0, 0]]},
    }

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost", port=5000, polling_timeout=25, timeout_exception=False
    )

    # Submit the routing problem to the server
    solution = cuopt_service_client.get_optimized_routes(data)

    # Number of repoll requests to be carried out for a successful response
    repoll_tries = 500

    # Repoll if needed
    solution = repoll(cuopt_service_client, solution, repoll_tries)

    # Display the solution
    print(json.dumps(solution, indent=4))


if __name__ == "__main__":
    main()
