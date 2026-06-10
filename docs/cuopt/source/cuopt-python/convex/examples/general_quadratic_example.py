# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
General Convex Quadratic Constraint Example
===========================================

This example demonstrates adding a general convex quadratic constraint
with the cuOpt Python API.

Problem:
    minimize    x + y
    subject to  x + y >= -5
                2*x^2 + 2*x*y + 2*y^2 <= 6     (general convex quadratic)

The quadratic constraint is an ellipsoid (a convex set). The
linear objective is minimized on it at x = y = -1, objective -2.
"""

from cuopt.linear_programming.problem import (
    MINIMIZE,
    Problem,
)


def main():
    prob = Problem("General Convex QC")

    x = prob.addVariable(lb=-float("inf"), name="x")
    y = prob.addVariable(lb=-float("inf"), name="y")

    # A linear constraint
    prob.addConstraint(x + y >= -5)

    # General convex quadratic constraint 2*x^2 + 2*x*y + 2*y^2 <= 6.
    prob.addConstraint(
        2 * x * x + 2 * x * y + 2 * y * y <= 6, name="ellipsoid"
    )

    prob.setObjective(x + y, sense=MINIMIZE)

    # cuOpt automatically selects the barrier method for problems with quadratic constraints.
    prob.solve()

    print(f"Status: {prob.Status}")
    print(f"x = {x.Value}")
    print(f"y = {y.Value}")
    print(f"Objective value = {prob.ObjValue}")


if __name__ == "__main__":
    main()
