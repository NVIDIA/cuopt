# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Simple Quadratic Programming Example
====================================

.. note::
   The QP solver is currently in beta.

This example demonstrates how to formulate and solve a simple
Quadratic Programming (QP) problem using the cuOpt Python API.

Problem:
    minimize    x^2 + y^2
    subject to  x + y >= 1
                x, y >= 0

This is a convex QP that minimizes the squared distance from the origin
while requiring the sum of x and y to be at least 1.
"""

from cuopt.linear_programming.problem import (
    MINIMIZE,
    Problem,
)


def main():
    # Create a new optimization problem
    prob = Problem("Simple QP")

    # Add variables with non-negative bounds
    x = prob.addVariable(lb=0, name="x")
    y = prob.addVariable(lb=0, name="y")

    # Add constraint: x + y >= 1
    prob.addConstraint(x + y >= 1)

    # Set quadratic objective: minimize x^2 + y^2
    # Using Variable * Variable to create quadratic terms
    quad_obj = x * x + y * y
    prob.setObjective(quad_obj, sense=MINIMIZE)

    # Solve the problem
    prob.solve()

    # Print results
    print(f"Optimal solution found in {prob.SolveTime:.2f} seconds")
    print(f"x = {x.Value}")
    print(f"y = {y.Value}")
    print(f"Objective value = {prob.ObjValue}")


if __name__ == "__main__":
    main()
