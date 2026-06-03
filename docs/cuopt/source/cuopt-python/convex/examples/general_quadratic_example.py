# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
General Convex Quadratic Constraint Example
===========================================

This example demonstrates a general convex quadratic constraint
``x^T Q x + d^T x <= alpha`` with the cuOpt Python API. Unlike a second-order
cone written in normal form, ``Q`` may be any matrix whose symmetric part
``(Q + Q^T)/2`` is positive semidefinite, the constraint may have a nonzero
right-hand side, and the cross term need not be supplied symmetrically — cuOpt
symmetrizes ``Q`` internally and converts the constraint to a second-order cone.

Problem:
    minimize    x + y
    subject to  x + y >= -5
                2*x^2 + 2*x*y + 2*y^2 <= 6     (general convex quadratic)

The quadratic constraint is an ellipsoid (its matrix is positive definite). The
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

    # A linear constraint (a problem with only a quadratic constraint is not
    # supported; include at least one linear row).
    prob.addConstraint(x + y >= -5)

    # General convex quadratic constraint 2*x^2 + 2*x*y + 2*y^2 <= 6. The cross
    # term is given naturally as a single 2*x*y; cuOpt symmetrizes Q internally.
    prob.addConstraint(
        2 * x * x + 2 * x * y + 2 * y * y <= 6, name="ellipsoid"
    )

    prob.setObjective(x + y, sense=MINIMIZE)

    # cuOpt automatically selects the barrier method for quadratic constraints.
    prob.solve()

    print(f"Status: {prob.Status}")
    print(f"x = {x.Value}")
    print(f"y = {y.Value}")
    print(f"Objective value = {prob.ObjValue}")


if __name__ == "__main__":
    main()
