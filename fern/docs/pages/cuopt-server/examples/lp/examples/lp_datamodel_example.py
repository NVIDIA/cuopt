# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LP DataModel from LP file parser example

This example demonstrates how to:
- Read an LP-format file using cuopt.linear_programming.Read
- Create a DataModel from the parsed LP file
- Solve using the DataModel via the server
- Extract detailed solution information

Requirements:
    - cuOpt server running (default: localhost:5000)
    - cuopt_sh_client package installed
    - cuopt package installed

Problem (in LP format; same instance as the MPS datamodel example):
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
from cuopt.linear_programming import Read
import time


def main():
    """Run the LP file DataModel example."""
    data = "sample.lp"

    lp_data = r"""\ Same problem as mps_datamodel_example.py (good-1)
Minimize
  -0.2 VAR1 + 0.1 VAR2
Subject To
  ROW1: 3 VAR1 + 4 VAR2 <= 5.4
  ROW2: 2.7 VAR1 + 10.1 VAR2 <= 4.9
End
"""

    with open(data, "w") as file:
        file.write(lp_data)

    print(f"Created LP file: {data}")

    print("\n=== Parsing LP File ===")
    parse_start = time.time()
    data_model = Read(data)
    parse_time = time.time() - parse_start
    print(f"Parse time: {parse_time:.3f} seconds")

    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost", port=5000, timeout_exception=False
    )

    ss = ThinClientSolverSettings()
    ss.set_parameter("pdlp_solver_mode", PDLPSolverMode.Fast1)
    ss.set_optimality_tolerance(1e-4)
    ss.set_parameter("time_limit", 5)

    print("\n=== Solving with Server ===")
    network_time = time.time()
    solution = cuopt_service_client.get_LP_solve(data_model, ss)
    network_time = time.time() - network_time

    solution_status = solution["response"]["solver_response"]["status"]
    solution_obj = solution["response"]["solver_response"]["solution"]

    print("\n=== Results ===")
    print(f"Termination Reason: {solution_status}")
    print(f"Objective Value: {solution_obj.get_primal_objective()}")
    print(f"LP Parse time: {parse_time:.3f} sec")

    network_time = network_time - (solution_obj.get_solve_time())
    print(f"Network time: {network_time:.3f} sec")

    solve_time = solution_obj.get_solve_time()
    print(f"Engine Solve time: {solve_time:.3f} sec")

    end_to_end_time = parse_time + network_time + solve_time
    print(f"Total end to end time: {end_to_end_time:.3f} sec")
    print(f"Variables Values: {solution_obj.get_vars()}")


if __name__ == "__main__":
    main()
