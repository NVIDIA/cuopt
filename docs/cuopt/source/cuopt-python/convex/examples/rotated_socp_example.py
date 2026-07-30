# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rotated Second-Order Cone Programming (SOCP) Example
====================================================

This example formulates and solves a rotated second-order cone problem with the
cuOpt Python API.

Problem:
    minimize    x3 + x4
    subject to  x1 + x2 >= 2
                x1^2 + x2^2 <= x3 * x4     (rotated second-order cone)
                x3 >= 0, x4 >= 0

The rotated cone is written as ``x1^2 + x2^2 - x3*x4 <= 0``.

Optimal solution: x1 = x2 = 1, x3 = x4 = sqrt(2) ~= 1.4142, objective ~= 2.8284.
"""

from cuopt.linear_programming.problem import (
    MINIMIZE,
    Problem,
)


def main():
    prob = Problem("Rotated SOCP")

    # Cone tail variables can be free; the cone heads must be non-negative.
    x1 = prob.addVariable(lb=-float("inf"), name="x1")
    x2 = prob.addVariable(lb=-float("inf"), name="x2")
    x3 = prob.addVariable(lb=0, name="x3")
    x4 = prob.addVariable(lb=0, name="x4")

    # Linear constraint
    prob.addConstraint(x1 + x2 >= 2)

    # x1^2 + x2^2 <= x3*x4
    prob.addConstraint(
        x1 * x1 + x2 * x2 - x3 * x4 <= 0,
        name="rotated_soc",
    )

    prob.setObjective(x3 + x4, sense=MINIMIZE)

    # cuOpt automatically selects the barrier method for quadratic constraints.
    prob.solve()

    print(f"Status: {prob.Status}")
    print(f"x1 = {x1.Value}")
    print(f"x2 = {x2.Value}")
    print(f"x3 = {x3.Value}")
    print(f"x4 = {x4.Value}")
    print(f"Objective value = {prob.ObjValue}")


if __name__ == "__main__":
    main()
