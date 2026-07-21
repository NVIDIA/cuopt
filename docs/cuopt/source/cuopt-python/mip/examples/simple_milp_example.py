# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mixed Integer Linear Programming Example

This example demonstrates how to:
- Create a mixed integer programming problem
- Add integer variables with bounds
- Add constraints with integer variables
- Solve a MIP problem

Problem:
    Maximize: 5*x + 3*y
    Subject to:
        2*x + 4*y >= 230
        3*x + 2*y <= 190
        10 <= y <= 50
        x, y are integers

Expected Output:
    Optimal solution found in 0.00 seconds
    x = 36.0
    y = 41.0
    Objective value = 303.0
"""

from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings


def main():
    """Run the simple MIP example."""
    # Create a new MIP problem
    problem = Problem("Simple MIP")

    # Add integer variables with bounds
    x = problem.addVariable(vtype=INTEGER, name="V_x")
    y = problem.addVariable(lb=10, ub=50, vtype=INTEGER, name="V_y")

    # Add constraints
    problem.addConstraint(2 * x + 4 * y >= 230, name="C1")
    problem.addConstraint(3 * x + 2 * y <= 190, name="C2")

    # Set objective function
    problem.setObjective(5 * x + 3 * y, sense=MAXIMIZE)

    # Configure solver settings
    settings = SolverSettings()
    settings.set_parameter("time_limit", 60)

    # Solve the problem
    problem.solve(settings)

    # Check solution status and results
    if problem.Status.name == "Optimal":
        print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"Objective value = {problem.ObjValue}")
    else:
        print(f"Problem status: {problem.Status.name}")


if __name__ == "__main__":
    main()
