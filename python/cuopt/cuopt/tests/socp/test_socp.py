# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Barrier SOCP / QCQP tests via the Problem Python API.

Python ``dual_api_`` tests exercise each SOC translation path through the Problem
API and verify dual extraction only: ``get_dual_solution()`` length/layout and
``Constraint.DualValue`` agree for linear and quadratic rows.
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
DUAL_TOL = 1e-5


def _barrier_settings() -> SolverSettings:
    settings = SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)
    return settings


def _quadratic_constraint_violation(constr, variables) -> float:
    """QCMATRIX row value minus rhs (feasible when <= 0 for L rows)."""
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


def _assert_solution_on_original_model(problem: Problem, solution) -> None:
    primal = solution.get_primal_solution()
    assert len(primal) == problem.NumVariables
    assert problem.ObjValue == pytest.approx(
        solution.get_primal_objective(), rel=0, abs=OBJ_TOL
    )
    assert problem.ObjValue == pytest.approx(
        problem.getObjective().getValue(), rel=0, abs=OBJ_TOL
    )


def _assert_dual_layout(problem: Problem, solution) -> None:
    """``get_dual_solution()`` matches per-constraint ``DualValue`` on the user model."""
    dual = solution.get_dual_solution()
    n_linear = sum(1 for c in problem.getConstraints() if not c.is_quadratic)
    n_qc = sum(1 for c in problem.getConstraints() if c.is_quadratic)
    assert len(dual) == n_linear + n_qc

    lin_idx = 0
    qc_idx = 0
    for constr in problem.getConstraints():
        if constr.is_quadratic:
            assert constr.DualValue == pytest.approx(
                dual[n_linear + qc_idx], rel=0, abs=DUAL_TOL
            )
            qc_idx += 1
        else:
            assert constr.DualValue == pytest.approx(
                dual[lin_idx], rel=0, abs=DUAL_TOL
            )
            lin_idx += 1


def _assert_duals_finite(problem: Problem) -> None:
    for constr in problem.getConstraints():
        assert np.isfinite(constr.DualValue)


def _solve_barrier(problem: Problem):
    solution = problem.solve(_barrier_settings())
    assert problem.Status.name == "Optimal"
    _assert_solution_on_original_model(problem, solution)
    _assert_feasible(problem)
    return solution


def _solve_and_assert_dual_api(problem: Problem) -> None:
    solution = _solve_barrier(problem)
    _assert_dual_layout(problem, solution)
    _assert_duals_finite(problem)
    assert any(c.is_quadratic for c in problem.getConstraints())


# ---------------------------------------------------------------------------
# Path-specific builders
# ---------------------------------------------------------------------------


def build_lorentz_path_problem() -> tuple[Problem, tuple]:
    """
    LORENTZ translation path (mirrors C++ ``qc_dual_recovery/lorentz_path``).

    min t
    s.t. x1 = 1, x2 = 0, x1^2 + x2^2 <= t^2, t >= 0.

    Optimum: t = 1, obj = 1, mu_QC = 1/2.
    """
    problem = Problem("path_lorentz")
    x1 = problem.addVariable(lb=-np.inf, name="x1")
    x2 = problem.addVariable(lb=-np.inf, name="x2")
    t = problem.addVariable(lb=0, name="t")
    problem.setObjective(t)
    problem.addConstraint(x1 == 1)
    problem.addConstraint(x2 == 0)
    problem.addConstraint(x1 * x1 + x2 * x2 - t * t <= 0, name="lorentz_soc")
    return problem, (x1, x2, t)


def build_lorentz_degenerate_problem() -> tuple[Problem, tuple]:
    """
    LORENTZ apex case (mirrors C++ ``qc_dual_recovery/lorentz_degenerate``).

    min t
    s.t. x1 = 0, x2 = 0, x1^2 + x2^2 <= t^2, t >= 0.

    Optimum: x1 = x2 = t = 0, obj = 0 at the cone apex.  grad g = 0 and the QC
    multiplier is not unique; recovery exports mu_QC = 0 by convention.
    """
    problem = Problem("path_lorentz_degenerate")
    x1 = problem.addVariable(lb=-np.inf, name="x1")
    x2 = problem.addVariable(lb=-np.inf, name="x2")
    t = problem.addVariable(lb=0, name="t")
    problem.setObjective(t)
    problem.addConstraint(x1 == 0)
    problem.addConstraint(x2 == 0)
    problem.addConstraint(x1 * x1 + x2 * x2 - t * t <= 0, name="lorentz_soc")
    return problem, (x1, x2, t)


def build_affine_path_problem() -> tuple[Problem, tuple]:
    """
    AFFINE translation path (mirrors C++ ``qc_dual_recovery/affine_path``).

    min x2
    s.t. x0 = 1, x2^2 + x3^2 <= x0.

    Optimum: x0 = 1, x2 = -1, x3 = 0, obj = -1, mu_QC = 1/2.
    (C++ names the objective tail x1 and uses x1^2 + x2^2 <= x0; same model.)
    """
    problem = Problem("path_affine")
    x0 = problem.addVariable(lb=-np.inf, name="x0")
    x2 = problem.addVariable(lb=-np.inf, name="x2")
    x3 = problem.addVariable(lb=-np.inf, name="x3")
    problem.setObjective(x2)
    problem.addConstraint(x0 == 1)
    problem.addConstraint(x2 * x2 + x3 * x3 - x0 <= 0, name="affine_soc")
    return problem, (x0, x2, x3)


def build_rotated_path_problem() -> tuple[Problem, tuple]:
    """
    ROTATED translation path with free heads in the objective
    (mirrors C++ ``qc_dual_recovery/rotated_path``).

    min x0 + 3*x2
    s.t. x1 = 2, x3 = 0, x2^2 + x3^2 <= x0*x1, x0, x1 >= 0.

    Active QC: x2^2 = 2*x0 => x2 = -sqrt(2*x0).  Minimizing
    f(x0) = x0 - 3*sqrt(2*x0) gives x0 = 9/2, x2 = -3, obj = -9/2.

    KKT (lambda on g = x2^2 + x3^2 - x0*x1 <= 0):
      x0: 1 - lambda*x1 = 0  =>  lambda = 1/2
      x2: 3 + lambda*2*x2 = 0
      EQ x1: DualValue = -lambda*x0 = -9/4;  EQ x3: 0.

    At the optimum the lifted slacks are s0 = h*(x0+x1) > 0 and s1 = h*(x0-x1) > 0
    with h = 1/2, so recovery uses the rotated path with nonzero s1.
    """
    problem = Problem("path_rotated")
    x0 = problem.addVariable(lb=0, name="x0")
    x1 = problem.addVariable(lb=0, name="x1")
    x2 = problem.addVariable(lb=-np.inf, name="x2")
    x3 = problem.addVariable(lb=-np.inf, name="x3")
    problem.setObjective(x0 + 3 * x2)
    problem.addConstraint(x1 == 2)
    problem.addConstraint(x3 == 0)
    problem.addConstraint(
        x2 * x2 + x3 * x3 - 0.5 * x0 * x1 - 0.5 * x1 * x0 <= 0,
        name="rotated_soc",
    )
    return problem, (x0, x1, x2, x3)


def build_rotated_degenerate_problem() -> tuple[Problem, tuple]:
    """
    ROTATED apex case (mirrors C++ ``qc_dual_recovery/rotated_degenerate``).

    min x0 + x1
    s.t. x0 = 0, x1 = 0, x2^2 + x3^2 <= x0*x1, x0, x1 >= 0.

    Optimum: all variables zero, obj = 0 at the lifted cone apex.  Both rotated
    heads are pinned to zero (s0 = s1 = 0), grad g = 0, and the QC multiplier
    is not unique; recovery exports mu_QC = 0 by convention.
    """
    problem = Problem("path_rotated_degenerate")
    x0 = problem.addVariable(lb=0, name="x0")
    x1 = problem.addVariable(lb=0, name="x1")
    x2 = problem.addVariable(lb=-np.inf, name="x2")
    x3 = problem.addVariable(lb=-np.inf, name="x3")
    problem.setObjective(x0 + x1)
    problem.addConstraint(x0 == 0)
    problem.addConstraint(x1 == 0)
    problem.addConstraint(
        x2 * x2 + x3 * x3 - 0.5 * x0 * x1 - 0.5 * x1 * x0 <= 0,
        name="rotated_soc",
    )
    return problem, (x0, x1, x2, x3)


def build_general_path_problem() -> tuple[Problem, tuple]:
    """
    GENERAL (LDLT) translation path, full-rank PSD QC
    (mirrors C++ ``qc_dual_recovery/general_path``).

    min 2*x0 + x1
    s.t. 2*x0^2 + 2*x0*x1 + 2*x1^2 <= 1, x0 - 3*x1 = 0.

    Optimum: x0 = -3/sqrt(26), x1 = -1/sqrt(26), obj = -7/sqrt(26),
    mu_QC = 7/(2*sqrt(26)), y_KKT = -3/26 on the equality row.
    """
    problem = Problem("path_general")
    x0 = problem.addVariable(lb=-np.inf, name="x0")
    x1 = problem.addVariable(lb=-np.inf, name="x1")
    problem.setObjective(2 * x0 + x1)
    problem.addConstraint(
        2 * x0 * x0 + 2 * x0 * x1 + 2 * x1 * x1 - 1 <= 0, name="general_qc"
    )
    problem.addConstraint(x0 - 3 * x1 == 0)
    return problem, (x0, x1)


def test_socp_dual_api_lorentz_path():
    problem, _ = build_lorentz_path_problem()
    _solve_and_assert_dual_api(problem)


def test_socp_dual_api_lorentz_degenerate():
    problem, _ = build_lorentz_degenerate_problem()
    _solve_and_assert_dual_api(problem)


def test_socp_dual_api_affine_path():
    problem, _ = build_affine_path_problem()
    _solve_and_assert_dual_api(problem)


def test_socp_dual_api_rotated_degenerate():
    problem, _ = build_rotated_degenerate_problem()
    _solve_and_assert_dual_api(problem)


def test_socp_dual_api_rotated_path():
    problem, _ = build_rotated_path_problem()
    _solve_and_assert_dual_api(problem)


def test_socp_dual_api_general_path():
    problem, _ = build_general_path_problem()
    _solve_and_assert_dual_api(problem)


# ---------------------------------------------------------------------------
# Larger Lorentz-style regression models (primal only)
# ---------------------------------------------------------------------------


def _soc_two_dim_constraint(problem, x0, x1, mat, head) -> None:
    z0 = problem.addVariable(lb=-np.inf)
    z1 = problem.addVariable(lb=-np.inf)
    problem.addConstraint(z0 == mat[0, 0] * x0 + mat[0, 1] * x1)
    problem.addConstraint(z1 == mat[1, 0] * x0 + mat[1, 1] * x1)
    problem.addConstraint(z0 * z0 + z1 * z1 - head * head <= 0)


def build_socp_1() -> tuple[Problem, tuple]:
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


def test_socp_1_barrier_solution():
    problem, (x0, x1, x2, y) = build_socp_1()
    _solve_barrier(problem)
    assert problem.ObjValue == pytest.approx(
        EXPECTED_SOCP_1_OBJECTIVE, abs=OBJ_TOL
    )
    assert x0.Value == pytest.approx(EXPECTED_SOCP_1_X[0], abs=PRIMAL_TOL)
    assert x1.Value == pytest.approx(EXPECTED_SOCP_1_X[1], abs=PRIMAL_TOL)
    assert x2.Value == pytest.approx(EXPECTED_SOCP_1_X[2], abs=PRIMAL_TOL)
    assert y.Value == pytest.approx(EXPECTED_SOCP_1_Y, abs=PRIMAL_TOL)


def test_socp_3_barrier_solution():
    problem, (x0, x1, h1, h2, h3) = build_socp_3()
    _solve_barrier(problem)
    assert problem.ObjValue == pytest.approx(
        EXPECTED_SOCP_3_OBJECTIVE, abs=OBJ_TOL
    )
    assert x0.Value == pytest.approx(EXPECTED_SOCP_3_X[0], abs=PRIMAL_TOL)
    assert x1.Value == pytest.approx(EXPECTED_SOCP_3_X[1], abs=PRIMAL_TOL)
    assert h1.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
    assert h2.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
    assert h3.Value == pytest.approx(1.0, abs=PRIMAL_TOL)


def test_maximize_with_quadratic_constraint():
    from cuopt.linear_programming.problem import MAXIMIZE, MINIMIZE

    prob_min = Problem("qc_maximize_min")
    x = prob_min.addVariable(lb=-np.inf, name="x")
    y = prob_min.addVariable(lb=-np.inf, name="y")
    prob_min.addConstraint(x + y <= 10)
    prob_min.addConstraint(2 * x * x + 2 * x * y + 2 * y * y <= 6)
    prob_min.setObjective(x + y, sense=MINIMIZE)
    _solve_barrier(prob_min)
    assert prob_min.ObjValue == pytest.approx(-2.0, abs=OBJ_TOL)
    assert x.Value == pytest.approx(-1.0, abs=PRIMAL_TOL)
    assert y.Value == pytest.approx(-1.0, abs=PRIMAL_TOL)

    prob_max = Problem("qc_maximize_max")
    x = prob_max.addVariable(lb=-np.inf, name="x")
    y = prob_max.addVariable(lb=-np.inf, name="y")
    prob_max.addConstraint(x + y <= 10)
    prob_max.addConstraint(2 * x * x + 2 * x * y + 2 * y * y <= 6)
    prob_max.setObjective(x + y, sense=MAXIMIZE)
    _solve_barrier(prob_max)
    assert prob_max.ObjValue == pytest.approx(2.0, abs=OBJ_TOL)
    assert x.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
    assert y.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
