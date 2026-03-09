# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Problem Solution API Example

This example demonstrates how to use the Problem API to access the solution
after solve(): getSolution() and getIncumbentValues().

- getSolution() returns the Solution object from the last solve (None before
  solve or after the problem is modified).
- getIncumbentValues(solution, vars) returns the primal values for the
  given variables from that solution. Use it with getSolution() and
  getVariables() to get values in a list, or for a subset of variables.

Problem:
    Maximize: x + y
    Subject to:
        x + y <= 10
        x - y >= 0
        x, y >= 0

Expected Output:
    Optimal solution found in 0.01 seconds
    Objective: 10.0
    Values via var.Value: x=10.0, y=0.0
    Values via getIncumbentValues: [10.0, 0.0]
    Subset (x only) via getIncumbentValues: [10.0]
"""

from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings


def main():
    """Run the Problem solution API example."""
    problem = Problem("Solution API Example")

    x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
    y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

    problem.addConstraint(x + y <= 10, name="c1")
    problem.addConstraint(x - y >= 0, name="c2")
    problem.setObjective(x + y, sense=MAXIMIZE)

    settings = SolverSettings()
    problem.solve(settings)

    if problem.Status.name == "Optimal":
        print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
        print(f"Objective: {problem.ObjValue}")

        # Access values via variable attributes (populated by solve)
        print(
            f"Values via var.Value: x={x.getValue()}, y={y.getValue()}"
        )

        # Access solution via getSolution() and getIncumbentValues()
        solution = problem.getSolution()
        vars_list = problem.getVariables()
        values = problem.getIncumbentValues(solution, vars_list)
        print(f"Values via getIncumbentValues: {values}")

        # getIncumbentValues works with a subset of variables too
        values_x_only = problem.getIncumbentValues(solution, [x])
        print(f"Subset (x only) via getIncumbentValues: {values_x_only}")
    else:
        print(f"Problem status: {problem.Status.name}")


if __name__ == "__main__":
    main()
