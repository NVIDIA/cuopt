# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Semi-continuous Variable Example

This example demonstrates how to:
- Add a semi-continuous variable
- Solve a small MIP that uses the semi-continuous domain

Problem:
    Minimize: x
    Subject to:
        x + y = 1
        x is either 0 or in [5, 10]
        0 <= y <= 1

Expected Output:
    Optimal solution found in 0.00 seconds
    x = 0.0
    y = 1.0
    Objective value = 0.0
"""

from cuopt.linear_programming.problem import (
    CONTINUOUS,
    MINIMIZE,
    SEMI_CONTINUOUS,
    Problem,
)
from cuopt.linear_programming.solver_settings import SolverSettings


def main():
    """Run the semi-continuous variable example."""
    problem = Problem("Semi-continuous")

    x = problem.addVariable(lb=5.0, ub=10.0, vtype=SEMI_CONTINUOUS, name="x")
    y = problem.addVariable(lb=0.0, ub=1.0, vtype=CONTINUOUS, name="y")

    problem.addConstraint(x + y == 1.0)
    problem.setObjective(x, sense=MINIMIZE)

    settings = SolverSettings()
    settings.set_parameter("time_limit", 10)

    problem.solve(settings)

    if problem.Status.name == "Optimal":
        print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"Objective value = {problem.ObjValue}")
    else:
        print(f"Problem status: {problem.Status.name}")


if __name__ == "__main__":
    main()
