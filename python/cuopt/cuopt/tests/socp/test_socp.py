# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Barrier SOCP tests via the Problem Python API.

Checks that the barrier solution is mapped back to the original model variables
after SOC conversion (see ``project_barrier_solution_to_model_variables`` in
``cpp/src/barrier/translate_soc.hpp``).
"""

from __future__ import annotations

import numpy as np
import pytest

from cuopt.linear_programming.problem import EQ, GE, LE, Problem
from cuopt.linear_programming.solver.solver_parameters import CUOPT_METHOD
from cuopt.linear_programming.solver_settings import (
    SolverMethod,
    SolverSettings,
)

EXPECTED_SOCP_1_OBJECTIVE = -13.548638904065102
EXPECTED_SOCP_1_X = (-3.874621860638774, -2.129788233677883, 2.33480343377204)
EXPECTED_SOCP_1_Y = 5.0

EXPECTED_SOCP_3_OBJECTIVE = -1.932105
EXPECTED_SOCP_3_X = (0.83666003, -0.54772256)

OBJ_TOL = 1e-6
PRIMAL_TOL = 1e-6
FEAS_TOL = 1e-6


def _barrier_settings() -> SolverSettings:
    settings = SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)
    return settings


def _soc_two_dim_constraint(problem, x0, x1, mat, head) -> None:
    """Encode ||mat @ [x0, x1]||_2 <= head as a standard Lorentz cone in (head, z0, z1)."""
    z0 = problem.addVariable(lb=-np.inf)
    z1 = problem.addVariable(lb=-np.inf)
    problem.addConstraint(z0 == mat[0, 0] * x0 + mat[0, 1] * x1)
    problem.addConstraint(z1 == mat[1, 0] * x0 + mat[1, 1] * x1)
    problem.addConstraint(z0 * z0 + z1 * z1 - head * head <= 0)


def build_socp_1() -> tuple[Problem, tuple]:
    """Min 3*x0+2*x1+x2  s.t. ||x||_2 <= y, x0+x1+3*x2 >= 1, 0 <= y <= 5."""
    problem = Problem("socp_1")
    x0 = problem.addVariable(lb=-np.inf, name="x0")
    x1 = problem.addVariable(lb=-np.inf, name="x1")
    x2 = problem.addVariable(lb=-np.inf, name="x2")
    y = problem.addVariable(lb=0, name="y")
    problem.setObjective(3 * x0 + 2 * x1 + x2)
    problem.addConstraint(y >= 0)
    problem.addConstraint(x0 * x0 + x1 * x1 + x2 * x2 - y * y <= 0)
    problem.addConstraint(x0 + x1 + 3 * x2 >= 1)
    problem.addConstraint(y <= 5)
    return problem, (x0, x1, x2, y)


def build_socp_3() -> tuple[Problem, tuple]:
    """Min -x0+2*x1  s.t. ||M_i x||_2 <= 1  for three fixed 2x2 maps M_i."""
    root2 = np.sqrt(2.0)
    u = np.array([[1 / root2, -1 / root2], [1 / root2, 1 / root2]])
    mat1 = np.diag([root2, 1 / root2]) @ u.T
    mat2 = np.diag([1.0, 1.0])
    mat3 = np.diag([0.2, 1.8])

    problem = Problem("socp_3")
    x0 = problem.addVariable(lb=-np.inf, name="x0")
    x1 = problem.addVariable(lb=-np.inf, name="x1")
    problem.setObjective(-x0 + 2 * x1)
    h1 = problem.addVariable(lb=1, ub=1, name="h1")
    h2 = problem.addVariable(lb=1, ub=1, name="h2")
    h3 = problem.addVariable(lb=1, ub=1, name="h3")
    _soc_two_dim_constraint(problem, x0, x1, mat1, h1)
    _soc_two_dim_constraint(problem, x0, x1, mat2, h2)
    _soc_two_dim_constraint(problem, x0, x1, mat3, h3)
    return problem, (x0, x1, h1, h2, h3)


def _quadratic_constraint_violation(constr, variables) -> float:
    """QCMATRIX row value minus rhs (should be <= 0 for L rows)."""
    vals = [var.Value for var in variables]
    quad = 0.0
    for k in range(len(constr.vals)):
        i = int(constr.rows[k])
        j = int(constr.cols[k])
        quad += float(constr.vals[k]) * vals[i] * vals[j]
    lin = 0.0
    for k in range(len(constr.linear_values)):
        lin += (
            float(constr.linear_values[k])
            * vals[int(constr.linear_indices[k])]
        )
    return quad + lin - float(constr.rhs_value)


def _assert_solution_on_original_model(problem: Problem, solution) -> None:
    primal = solution.get_primal_solution()
    assert len(primal) == problem.NumVariables
    assert problem.ObjValue == pytest.approx(
        solution.get_primal_objective(), rel=0, abs=OBJ_TOL
    )
    assert problem.ObjValue == pytest.approx(
        problem.getObjective().getValue(), rel=0, abs=OBJ_TOL
    )


def _assert_feasible(problem: Problem) -> None:
    variables = problem.getVariables()
    for constr in problem.getConstraints():
        if constr.is_quadratic:
            assert (
                _quadratic_constraint_violation(constr, variables) <= FEAS_TOL
            )
            continue
        slack = constr.compute_slack()
        if constr.Sense == LE:
            assert slack >= -FEAS_TOL
        elif constr.Sense == GE:
            assert slack <= FEAS_TOL
        else:
            assert constr.Sense == EQ
            assert slack == pytest.approx(0.0, abs=FEAS_TOL)


def _solve(problem: Problem):
    solution = problem.solve(_barrier_settings())
    assert problem.Status.name == "Optimal"
    return solution


def test_socp_1_barrier_solution():
    problem, (x0, x1, x2, y) = build_socp_1()
    solution = _solve(problem)
    _assert_solution_on_original_model(problem, solution)
    _assert_feasible(problem)

    assert problem.ObjValue == pytest.approx(
        EXPECTED_SOCP_1_OBJECTIVE, abs=OBJ_TOL
    )
    assert x0.Value == pytest.approx(EXPECTED_SOCP_1_X[0], abs=PRIMAL_TOL)
    assert x1.Value == pytest.approx(EXPECTED_SOCP_1_X[1], abs=PRIMAL_TOL)
    assert x2.Value == pytest.approx(EXPECTED_SOCP_1_X[2], abs=PRIMAL_TOL)
    assert y.Value == pytest.approx(EXPECTED_SOCP_1_Y, abs=PRIMAL_TOL)


def test_socp_3_barrier_solution():
    problem, (x0, x1, h1, h2, h3) = build_socp_3()
    solution = _solve(problem)
    _assert_solution_on_original_model(problem, solution)
    _assert_feasible(problem)

    assert problem.ObjValue == pytest.approx(
        EXPECTED_SOCP_3_OBJECTIVE, abs=OBJ_TOL
    )
    assert x0.Value == pytest.approx(EXPECTED_SOCP_3_X[0], abs=PRIMAL_TOL)
    assert x1.Value == pytest.approx(EXPECTED_SOCP_3_X[1], abs=PRIMAL_TOL)
    assert h1.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
    assert h2.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
    assert h3.Value == pytest.approx(1.0, abs=PRIMAL_TOL)


def test_general_quadratic_unsymmetric():
    """
    Min x0 + x1
    s.t. 2*x0^2 + 3*x0*x1 + 2*x1^2 <= 1  (unsymmetric Q: cross term only as x0*x1)
         x0 - x1 = 0

    Q is given unsymmetrically: the 3*x0*x1 term is stored as a single
    entry (row=0, col=1, val=3) rather than symmetric (0,1,1.5)+(1,0,1.5).
    After symmetrization H = [4 3; 3 4], eigenvalues 1 and 7 (PD).

    With x0 = x1 = t: 2t^2 + 3t^2 + 2t^2 = 7t^2 <= 1
    min 2t at t = -1/sqrt(7), obj = -2/sqrt(7) ≈ -0.755929
    """
    problem = Problem("general_qc_unsymmetric")
    x0 = problem.addVariable(lb=-np.inf, name="x0")
    x1 = problem.addVariable(lb=-np.inf, name="x1")
    problem.setObjective(x0 + x1)
    problem.addConstraint(2 * x0 * x0 + 3 * x0 * x1 + 2 * x1 * x1 <= 1)
    problem.addConstraint(x0 - x1 == 0)

    solution = _solve(problem)
    _assert_solution_on_original_model(problem, solution)
    _assert_feasible(problem)

    expected_obj = -2.0 / np.sqrt(7.0)
    expected_x = -1.0 / np.sqrt(7.0)
    assert problem.ObjValue == pytest.approx(expected_obj, abs=OBJ_TOL)
    assert x0.Value == pytest.approx(expected_x, abs=PRIMAL_TOL)
    assert x1.Value == pytest.approx(expected_x, abs=PRIMAL_TOL)


def test_maximize_with_quadratic_constraint():
    """
    Maximize x + y
    s.t.  x + y <= 10
          2*x^2 + 2*x*y + 2*y^2 <= 6

    The quadratic constraint is the binding one.
    With x = y = t: 2t^2 + 2t^2 + 2t^2 = 6t^2 <= 6 => t in [-1, 1].
    Maximizing 2t gives t = 1, obj = 2.

    Minimizing gives t = -1, obj = -2.

    This test verifies that MAXIMIZE is respected when quadratic constraints
    are present (regression for a bug where the QCQP path ignored the
    objective sense).
    """
    from cuopt.linear_programming.problem import MAXIMIZE, MINIMIZE

    # Solve as MINIMIZE first to establish baseline
    prob_min = Problem("qc_maximize_min")
    x = prob_min.addVariable(lb=-np.inf, name="x")
    y = prob_min.addVariable(lb=-np.inf, name="y")
    prob_min.addConstraint(x + y <= 10)
    prob_min.addConstraint(2 * x * x + 2 * x * y + 2 * y * y <= 6)
    prob_min.setObjective(x + y, sense=MINIMIZE)
    _solve(prob_min)
    _assert_feasible(prob_min)

    assert prob_min.ObjValue == pytest.approx(-2.0, abs=OBJ_TOL)
    assert x.Value == pytest.approx(-1.0, abs=PRIMAL_TOL)
    assert y.Value == pytest.approx(-1.0, abs=PRIMAL_TOL)

    # Solve as MAXIMIZE - should give the opposite optimum
    prob_max = Problem("qc_maximize_max")
    x = prob_max.addVariable(lb=-np.inf, name="x")
    y = prob_max.addVariable(lb=-np.inf, name="y")
    prob_max.addConstraint(x + y <= 10)
    prob_max.addConstraint(2 * x * x + 2 * x * y + 2 * y * y <= 6)
    prob_max.setObjective(x + y, sense=MAXIMIZE)
    _solve(prob_max)
    _assert_feasible(prob_max)

    assert prob_max.ObjValue == pytest.approx(2.0, abs=OBJ_TOL)
    assert x.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
    assert y.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
