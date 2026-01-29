# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Quadratic Programming Matrix Example
====================================

.. note::
   The QP solver is currently in beta.

This example demonstrates how to formulate and solve a
Quadratic Programming (QP) problem represented in a matrix format
using the cuOpt Python API.

Problem:
    minimize    0.01 * p1^2 + 0.02 * p2^2 + 0.015 * p3^2 + 8 * p1 + 6 * p2 + 7 * p3
    subject to  p1 + p2 + p3 = 150
                10 <= p1 <= 100
                10 <= p2 <= 80
                10 <= p3 <= 90

This is a convex QP that minimizes the cost of power generation and dispatch
while satisfying capacity and demand.
"""

from cuopt.linear_programming.problem import (
    MINIMIZE,
    Problem,
    QuadraticExpression,
)


def main():
    # Create a new optimization problem
    prob = Problem("QP Power Dispatch")

    # Add variables with lower and upper bounds
    p1 = prob.addVariable(lb=10, ub=100)
    p2 = prob.addVariable(lb=10, ub=80)
    p3 = prob.addVariable(lb=10, ub=90)

    # Add demand constraint: p1 + p2 + p3 = 150
    prob.addConstraint(p1 + p2 + p3 == 150, name="demand")

    # Create matrix for quadratic terms: 0.01 p1^2 + 0.02 p2^2 + 0.015 p3^2
    matrix = [[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.015]]
    quad_matrix = QuadraticExpression(matrix, prob.getVariables())

    # Set objective using matrix representation
    quad_obj = quad_matrix + 8 * p1 + 6 * p2 + 7 * p3
    prob.setObjective(quad_obj, sense=MINIMIZE)

    # Solve the problem
    prob.solve()

    # Print results
    print(f"Optimal solution found in {prob.SolveTime:.2f} seconds")
    print(f"p1 = {p1.Value}")
    print(f"p2 = {p2.Value}")
    print(f"p3 = {p3.Value}")
    print(f"Minimized cost = {prob.ObjValue}")


if __name__ == "__main__":
    main()
