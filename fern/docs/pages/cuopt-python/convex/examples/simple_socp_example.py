# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple Second-Order Cone Programming (SOCP) Example
===================================================

This example demonstrates how to formulate and solve a Second-Order Cone
Program (SOCP) using the cuOpt Python API.

Problem:
    minimize    x3
    subject to  x1 + x2 >= 2
                x1^2 + x2^2 <= x3^2     (i.e. ||(x1, x2)||_2 <= x3)
                x3 >= 0

The cone constraint is supplied as the quadratic inequality
``x1^2 + x2^2 - x3^2 <= 0``. cuOpt detects the second-order cone structure and
solves with the barrier method.

Optimal solution: x1 = x2 = 1, x3 = sqrt(2) ~= 1.4142, objective ~= 1.4142.
"""

from cuopt.linear_programming.problem import (
    MINIMIZE,
    Problem,
)


def main():
    prob = Problem("Simple SOCP")

    # Cone tail variables can be free; the cone head must be non-negative.
    x1 = prob.addVariable(lb=-float("inf"), name="x1")
    x2 = prob.addVariable(lb=-float("inf"), name="x2")
    x3 = prob.addVariable(lb=0, name="x3")

    # Linear constraint
    prob.addConstraint(x1 + x2 >= 2)

    # Second-order cone ||(x1, x2)||_2 <= x3 as x1^2 + x2^2 - x3^2 <= 0
    prob.addConstraint(x1 * x1 + x2 * x2 - x3 * x3 <= 0, name="soc")

    prob.setObjective(x3, sense=MINIMIZE)

    # cuOpt automatically selects the barrier method for quadratic constraints.
    prob.solve()

    print(f"Status: {prob.Status}")
    print(f"x1 = {x1.Value}")
    print(f"x2 = {x2.Value}")
    print(f"x3 = {x3.Value}")
    print(f"Objective value = {prob.ObjValue}")


if __name__ == "__main__":
    main()
