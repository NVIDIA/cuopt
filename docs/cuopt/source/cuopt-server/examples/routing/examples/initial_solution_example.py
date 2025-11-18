# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Initial Solution Routing Example

This example demonstrates how to use previous solutions as initial solutions
for new routing requests. This can help warm-start the solver and potentially
find better solutions faster.

Features demonstrated:
    - Using previous reqId as initial solution
    - Uploading a saved solution for reuse
    - Providing inline initial solution in data
    - Deleting saved solutions to free memory

Requirements:
    - cuOpt server running (default: localhost:5000)
    - cuopt_sh_client package installed

Note:
    Initial solutions may not always be accepted. For very small/simple problems,
    the solver might find the optimal solution immediately without needing the
    initial solution.
"""

from cuopt_sh_client import CuOptServiceSelfHostClient
import json


def main():
    """Run the initial solution routing example."""
    data = {
        "cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
        "task_data": {"task_locations": [0, 1]},
        "fleet_data": {"vehicle_locations": [[0, 0], [0, 0]]},
    }

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost", port=5000, timeout_exception=False
    )

    print("=== Getting Initial Solution ===")
    # Get initial solution
    # Set delete_solution to false so it can be used in next request
    initial_solution = cuopt_service_client.get_optimized_routes(
        data, delete_solution=False
    )

    print(f"Initial solution reqId: {initial_solution['reqId']}")

    print("\n=== Uploading Solution for Reuse ===")
    # Upload a solution returned/saved from previous request as initial solution
    initial_solution_3 = cuopt_service_client.upload_solution(initial_solution)

    print(f"Uploaded solution reqId: {initial_solution_3['reqId']}")

    print("\n=== Solving with Multiple Initial Solutions ===")
    # Use previous solution saved in server as initial solution to this request.
    # That solution is referenced with previous request id.
    solution = cuopt_service_client.get_optimized_routes(
        data,
        initial_ids=[initial_solution["reqId"], initial_solution_3["reqId"]],
    )

    print(json.dumps(solution, indent=4))

    # Delete saved solution if not required to save space
    print("\n=== Cleaning Up Saved Solutions ===")
    cuopt_service_client.delete(initial_solution["reqId"])
    cuopt_service_client.delete(initial_solution_3["reqId"])
    print("Saved solutions deleted")

    print("\n=== Using Inline Initial Solution ===")
    # Another option is to add a solution that was generated
    # to data model option as follows
    initial_solution_2 = [
        {
            "0": {
                "task_id": ["Depot", "0", "1", "Depot"],
                "type": ["Depot", "Delivery", "Delivery", "Depot"],
            }
        }
    ]

    data["initial_solution"] = initial_solution_2
    solution = cuopt_service_client.get_optimized_routes(data)

    print(json.dumps(solution, indent=4))

    print("\n=== Note ===")
    print("The initial solution in the response may show 'not accepted',")
    print("because the problem is too small and the optimal solution is")
    print("found even before cuOpt could use an initial solution.")


if __name__ == "__main__":
    main()
