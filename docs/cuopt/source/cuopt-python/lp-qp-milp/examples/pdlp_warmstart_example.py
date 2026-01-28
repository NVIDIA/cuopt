# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Working with PDLP Warmstart Data Example

This example demonstrates:
- Using PDLP (Primal-Dual hybrid gradient for LP) solver method
- Extracting warmstart data from a solved problem
- Reusing warmstart data to solve a similar problem faster
- Comparing solve times with and without warmstart

Warmstart data allows restarting PDLP with a previous solution context.
This should be used when you solve a new problem which is similar to the
previous one.

Note:
    Warmstart data is only available for Linear Programming (LP) problems,
    not for Mixed Integer Linear Programming (MILP) problems.

Problem 1:
    Maximize: 2*x + y
    Subject to:
        4*x + 10*y <= 130
        8*x - 3*y >= 40
        x, y >= 0

Problem 2 (similar):
    Maximize: 2*x + y
    Subject to:
        4*x + 10*y <= 100
        8*x - 3*y >= 50
        x, y >= 0

Expected Output:
    Optimal solution found in 0.01 seconds
    x = 25.0
    y = 0.0
    Objective value = 50.0
"""

from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver.solver_parameters import CUOPT_METHOD
from cuopt.linear_programming.solver_settings import (
    SolverSettings,
    SolverMethod,
)


def main():
    """Run the PDLP warmstart example."""
    print("=== Solving Problem 1 ===")

    # Create a new problem
    problem = Problem("Simple LP")

    # Add variables
    x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
    y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

    # Add constraints
    problem.addConstraint(4 * x + 10 * y <= 130, name="c1")
    problem.addConstraint(8 * x - 3 * y >= 40, name="c2")

    # Set objective function
    problem.setObjective(2 * x + y, sense=MAXIMIZE)

    # Configure solver settings
    settings = SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)

    # Solve the problem
    problem.solve(settings)

    print(f"Problem 1 solved in {problem.SolveTime:.4f} seconds")
    print(f"x = {x.getValue()}, y = {y.getValue()}")
    print(f"Objective value = {problem.ObjValue}")

    # Get the warmstart data
    warmstart_data = problem.getWarmstartData()
    print(
        f"\nWarmstart data extracted (primal solution size: "
        f"{len(warmstart_data.current_primal_solution)})"
    )

    print("\n=== Solving Problem 2 with Warmstart ===")

    # Create a new problem
    new_problem = Problem("Warmstart LP")

    # Add variables
    x = new_problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
    y = new_problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

    # Add constraints (slightly different from problem 1)
    new_problem.addConstraint(4 * x + 10 * y <= 100, name="c1")
    new_problem.addConstraint(8 * x - 3 * y >= 50, name="c2")

    # Set objective function
    new_problem.setObjective(2 * x + y, sense=MAXIMIZE)

    # Configure solver settings with warmstart data
    settings.set_pdlp_warm_start_data(warmstart_data)

    # Solve the problem
    new_problem.solve(settings)

    # Check solution status
    if new_problem.Status.name == "Optimal":
        print(f"Optimal solution found in {new_problem.SolveTime:.2f} seconds")
        print(f"x = {x.getValue()}")
        print(f"y = {y.getValue()}")
        print(f"Objective value = {new_problem.ObjValue}")
    else:
        print(f"Problem status: {new_problem.Status.name}")


if __name__ == "__main__":
    main()
