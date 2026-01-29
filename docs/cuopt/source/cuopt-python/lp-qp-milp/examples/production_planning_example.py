# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Production Planning Example

This example demonstrates a real-world application of MIP:
- Production planning with resource constraints
- Multiple constraint types (machine time, labor, materials)
- Profit maximization

Problem:
    A factory produces two products (A and B)
    - Product A: $50 profit per unit
    - Product B: $30 profit per unit

    Resources:
    - Machine time: 2 hrs/unit A, 1 hr/unit B, max 100 hrs
    - Labor: 1 hr/unit A, 3 hrs/unit B, max 120 hrs
    - Material: 4 units/unit A, 2 units/unit B, max 200 units

    Constraints:
    - Minimum 10 units of Product A
    - Minimum 15 units of Product B

Expected Output:
    === Production Planning Solution ===
    Status: Optimal
    Solve time: 0.09 seconds
    Product A production: 36.0 units
    Product B production: 28.0 units
    Total profit: $2640.00
"""

from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings


def main():
    """Run the production planning example."""
    # Production planning problem
    problem = Problem("Production Planning")

    # Decision variables: production quantities
    # x1 = units of product A
    # x2 = units of product B
    x1 = problem.addVariable(lb=10, vtype=INTEGER, name="Product_A")
    x2 = problem.addVariable(lb=15, vtype=INTEGER, name="Product_B")

    # Resource constraints
    # Machine time: 2 hours per unit of A, 1 hour per unit of B, max 100 hours
    problem.addConstraint(2 * x1 + x2 <= 100, name="Machine_Time")

    # Labor: 1 hour per unit of A, 3 hours per unit of B, max 120 hours
    problem.addConstraint(x1 + 3 * x2 <= 120, name="Labor_Hours")

    # Material: 4 units per unit of A, 2 units per unit of B, max 200 units
    problem.addConstraint(4 * x1 + 2 * x2 <= 200, name="Material")

    # Objective: maximize profit
    # Profit: $50 per unit of A, $30 per unit of B
    problem.setObjective(50 * x1 + 30 * x2, sense=MAXIMIZE)

    # Solve with time limit
    settings = SolverSettings()
    settings.set_parameter("time_limit", 30)
    problem.solve(settings)

    # Display results
    if problem.Status.name == "Optimal":
        print("=== Production Planning Solution ===")
        print(f"Status: {problem.Status.name}")
        print(f"Solve time: {problem.SolveTime:.2f} seconds")
        print(f"Product A production: {x1.getValue()} units")
        print(f"Product B production: {x2.getValue()} units")
        print(f"Total profit: ${problem.ObjValue:.2f}")

    else:
        print(f"Problem not solved optimally. Status: {problem.Status.name}")


if __name__ == "__main__":
    main()
