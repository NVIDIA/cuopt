# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LP DataModel from MPS Parser Example

This example demonstrates how to:
- Parse an MPS file using cuopt_mps_parser
- Create a DataModel from the parsed MPS
- Solve using the DataModel via the server
- Extract detailed solution information

Requirements:
    - cuOpt server running (default: localhost:5000)
    - cuopt_sh_client package installed
    - cuopt_mps_parser package installed

Problem (in MPS format):
    Minimize: -0.2*VAR1 + 0.1*VAR2
    Subject to:
        3*VAR1 + 4*VAR2 <= 5.4
        2.7*VAR1 + 10.1*VAR2 <= 4.9
        VAR1, VAR2 >= 0

Expected Output:
    Termination Reason: 1 (Optimal)
    Objective Value: -0.36
    Variables Values: {'VAR1': 1.8, 'VAR2': 0.0}
"""

from cuopt_sh_client import (
    CuOptServiceSelfHostClient,
    ThinClientSolverSettings,
    PDLPSolverMode,
)
import cuopt_mps_parser
import time


def main():
    """Run the MPS DataModel example."""
    # -- Parse the MPS file --

    data = "sample.mps"

    mps_data = """NAME   good-1
ROWS
 N  COST
 L  ROW1
 L  ROW2
COLUMNS
   VAR1      COST      -0.2
   VAR1      ROW1      3              ROW2      2.7
   VAR2      COST      0.1
   VAR2      ROW1      4              ROW2      10.1
RHS
   RHS1      ROW1      5.4            ROW2      4.9
ENDATA
"""

    with open(data, "w") as file:
        file.write(mps_data)

    print(f"Created MPS file: {data}")

    # Parse the MPS file and measure the time spent
    print("\n=== Parsing MPS File ===")
    parse_start = time.time()
    data_model = cuopt_mps_parser.ParseMps(data)
    parse_time = time.time() - parse_start
    print(f"Parse time: {parse_time:.3f} seconds")

    # -- Build the client object --

    # If cuOpt is not running on localhost:5000, edit `ip` and `port` parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost", port=5000, timeout_exception=False
    )

    # -- Set the solver settings --

    ss = ThinClientSolverSettings()

    # Set the solver mode to Fast1 (Stable1 could also be used)
    ss.set_parameter("pdlp_solver_mode", PDLPSolverMode.Fast1)

    # Set the general tolerance to 1e-4 (default value)
    ss.set_optimality_tolerance(1e-4)

    # Optional: Set iteration and time limits
    # ss.set_iteration_limit(1000)
    # ss.set_time_limit(10)
    ss.set_parameter("time_limit", 5)

    # -- Call solve --

    print("\n=== Solving with Server ===")
    network_time = time.time()
    solution = cuopt_service_client.get_LP_solve(data_model, ss)
    network_time = time.time() - network_time

    # -- Retrieve the solution object and print the details --

    solution_status = solution["response"]["solver_response"]["status"]
    solution_obj = solution["response"]["solver_response"]["solution"]

    print("\n=== Results ===")
    # Check Termination Reason
    print(f"Termination Reason: {solution_status}")

    # Check found objective value
    print(f"Objective Value: {solution_obj.get_primal_objective()}")

    # Check the MPS parse time
    print(f"MPS Parse time: {parse_time:.3f} sec")

    # Check network time (client call - solve time)
    network_time = network_time - (solution_obj.get_solve_time())
    print(f"Network time: {network_time:.3f} sec")

    # Check solver time
    solve_time = solution_obj.get_solve_time()
    print(f"Engine Solve time: {solve_time:.3f} sec")

    # Check the total end to end time (mps parsing + network + solve time)
    end_to_end_time = parse_time + network_time + solve_time
    print(f"Total end to end time: {end_to_end_time:.3f} sec")

    # Print the found decision variables
    print(f"Variables Values: {solution_obj.get_vars()}")


if __name__ == "__main__":
    main()
