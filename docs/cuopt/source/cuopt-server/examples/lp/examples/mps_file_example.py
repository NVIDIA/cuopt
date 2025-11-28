# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LP MPS File Server Example

This example demonstrates how to solve LP problems from MPS files using the
cuOpt server. MPS (Mathematical Programming System) is a standard file format
for representing optimization problems.

Requirements:
    - cuOpt server running (default: localhost:5000)
    - cuopt_sh_client package installed

Problem (in MPS format):
    Minimize: -0.2*VAR1 + 0.1*VAR2
    Subject to:
        3*VAR1 + 4*VAR2 <= 5.4
        2.7*VAR1 + 10.1*VAR2 <= 4.9
        VAR1, VAR2 >= 0

Expected Response:
    {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {
                    "primal_objective": -0.36,
                    "vars": {
                        "VAR1": 1.8,
                        "VAR2": 0.0
                    }
                }
            }
        }
    }
"""

from cuopt_sh_client import (
    CuOptServiceSelfHostClient,
    ThinClientSolverSettings,
)
import json
import os


def main():
    """Run the MPS file LP example."""
    data = "sample.mps"

    # MPS file content
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

    # Write MPS file
    with open(data, "w") as file:
        file.write(mps_data)

    print(f"Created MPS file: {data}")

    # If cuOpt is not running on localhost:5000, edit `ip` and `port` parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost", port=5000, timeout_exception=False
    )

    # Configure solver settings
    ss = ThinClientSolverSettings()
    ss.set_parameter("time_limit", 5)
    ss.set_optimality_tolerance(0.00001)

    print("\n=== Solving LP from MPS File ===")
    solution = cuopt_service_client.get_LP_solve(
        data, solver_config=ss, response_type="dict"
    )

    print(json.dumps(solution, indent=4))

    # Delete the mps file after solving
    if os.path.exists(data):
        os.remove(data)
        print(f"Deleted MPS file: {data}")


if __name__ == "__main__":
    main()
