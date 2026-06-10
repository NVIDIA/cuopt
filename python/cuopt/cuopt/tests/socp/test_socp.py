# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Barrier SOCP / QCQP tests via the Problem Python API.

Translation-path tests (``-k translation_``) check SOC conversion paths and that
recovered QC duals match closed-form KKT multipliers on the **user** model
(see ``cpp/src/barrier/translate_soc.hpp`` / ``project_barrier_qcqp_duals_to_model``).
Rotated and affine QCs use Lorentz-head recovery on the lifted block ``[s0, s1, tails]``
(``mu_user = mu_soc``).  Affine linking-row duals are separate equality multipliers.
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
KKT_TOL = 1e-4

# Closed-form optimal duals as exported by cuOpt on the user model (minimize).
#
# Standard KKT for min c'x s.t. h(x)=0, g(x)<=0 (active at optimum):
#   lambda_KKT >= 0 on g, y_KKT free on h,  c + y'∇h + lambda'∇g = 0.
#
# cuOpt Problem API / barrier export:
#   EQ row:  DualValue = -y_KKT
#   LE/QC row (g<=0): DualValue = lambda_KKT  (usual multiplier on g)
#
# Lorentz example: lambda_KKT = 1/(2t) = 1/2 at t=1.
_SQRT6 = float(np.sqrt(6.0))

TRANSLATION_EXPECTED_DUALS = {
    "lorentz": {
        "linear": [1.0, 0.0],  # x1=1, x2=0  (y_KKT = -1, 0)
        "qc": {"lorentz_soc": 0.5},
    },
    "affine": {
        "linear": [-0.5],  # x0=1  (y_KKT = +1/2)
        "qc": {"affine_soc": 0.5},
    },
    "rotated": {
        # x1=2, x3=0 (EQ); pi_x1 = -lambda*x0 = -9/4 at optimum
        "linear": [-2.25, 0.0],
        "qc": {"rotated_soc": 0.5},
    },
    "general": {
        "linear": [0.0],
        "qc": {"general_qc": 1.0 / _SQRT6},
    },
}


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


def _qc_gradient_component(constr, var, index_to_var) -> float:
    """Gradient of the QCMATRIX row (quadratic + linear parts) w.r.t. var."""
    j = var.index
    grad = 0.0
    for k in range(len(constr.vals)):
        r = int(constr.rows[k])
        c = int(constr.cols[k])
        v = float(constr.vals[k])
        xr = index_to_var[r].Value
        xc = index_to_var[c].Value
        if r == j and c == j:
            grad += 2.0 * v * xr
        elif r == j:
            grad += v * xc
        elif c == j:
            grad += v * xr
    for k in range(len(constr.linear_values)):
        if int(constr.linear_indices[k]) == j:
            grad += float(constr.linear_values[k])
    return grad


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
    dual = solution.get_dual_solution()
    n_linear = sum(1 for c in problem.getConstraints() if not c.is_quadratic)
    n_qc = sum(1 for c in problem.getConstraints() if c.is_quadratic)
    assert len(dual) == n_linear + n_qc
    qc_idx = 0
    for constr in problem.getConstraints():
        if not constr.is_quadratic:
            continue
        assert constr.DualValue == pytest.approx(
            dual[n_linear + qc_idx], rel=0, abs=DUAL_TOL
        )
        qc_idx += 1


def _assert_analytical_duals(
    problem: Problem,
    *,
    expected_linear: list[float],
    expected_qc: dict[str, float],
) -> None:
    linear = [c for c in problem.getConstraints() if not c.is_quadratic]
    assert len(linear) == len(expected_linear)
    for constr, expected in zip(linear, expected_linear):
        assert constr.DualValue == pytest.approx(expected, abs=DUAL_TOL), (
            f"linear constraint {constr.ConstraintName!r}"
        )

    qc_by_name = {
        c.ConstraintName: c for c in problem.getConstraints() if c.is_quadratic
    }
    for name, expected in expected_qc.items():
        assert qc_by_name[name].DualValue == pytest.approx(
            expected, abs=DUAL_TOL
        ), f"QC constraint {name!r}"


def _assert_qcqp_kkt(problem: Problem) -> None:
    """Stationarity and complementarity on the user QCQP using exported duals."""
    variables = problem.getVariables()
    index_to_var = {v.index: v for v in variables}
    residual = [v.Obj for v in variables]

    for constr in problem.getConstraints():
        pi = constr.DualValue
        if constr.is_quadratic:
            for var in variables:
                residual[var.index] += pi * _qc_gradient_component(
                    constr, var, index_to_var
                )
            viol = _quadratic_constraint_violation(constr, variables)
            assert pi * viol == pytest.approx(0.0, abs=KKT_TOL)
            assert pi >= -FEAS_TOL
        elif constr.Sense == EQ:
            for v_idx, coeff in constr.vindex_coeff_dict.items():
                residual[v_idx] -= pi * coeff
        elif constr.Sense == LE:
            for v_idx, coeff in constr.vindex_coeff_dict.items():
                residual[v_idx] += pi * coeff
        elif constr.Sense == GE:
            for v_idx, coeff in constr.vindex_coeff_dict.items():
                residual[v_idx] -= pi * coeff

    for j, r in enumerate(residual):
        assert r == pytest.approx(0.0, abs=KKT_TOL), (
            f"stationarity residual for variable index {j}: {r}"
        )


def _solve_barrier(problem: Problem):
    solution = problem.solve(_barrier_settings())
    assert problem.Status.name == "Optimal"
    _assert_solution_on_original_model(problem, solution)
    _assert_feasible(problem)
    return solution


def _solve_and_assert_translation_path(
    problem: Problem,
    path_key: str,
    *,
    expected_obj: float,
    var_expect: list[tuple],
) -> None:
    expected = TRANSLATION_EXPECTED_DUALS[path_key]
    solution = _solve_barrier(problem)
    _assert_dual_layout(problem, solution)
    _assert_analytical_duals(
        problem,
        expected_linear=expected["linear"],
        expected_qc=expected["qc"],
    )
    _assert_qcqp_kkt(problem)
    assert problem.ObjValue == pytest.approx(expected_obj, abs=OBJ_TOL)
    for var, expected_val in var_expect:
        assert var.Value == pytest.approx(expected_val, abs=PRIMAL_TOL)


# ---------------------------------------------------------------------------
# Path-specific builders
# ---------------------------------------------------------------------------


def build_lorentz_path_problem() -> tuple[Problem, tuple]:
    """Min t  s.t. x1=1, x2=0, x1^2+x2^2 <= t^2, t>=0  =>  t=1, lambda=1/2."""
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
    Min t  s.t. x1=0, x2=0, x1^2+x2^2 <= t^2, t>=0  =>  t=0 at the cone apex.

    The QC multiplier is not unique (grad g = 0); recovery exports mu = 0 by convention.
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
    """Min x2  s.t. x0=1, x2^2+x3^2 <= x0  =>  x2=-1, lambda=1/2."""
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
    Rotated SOC with a free head in the objective (not the old pinned-heads case).

    min x0 + 3*x2
    s.t. x1 = 2, x3 = 0, x2^2 + x3^2 <= x0*x1, x0,x1 >= 0.

    Active QC: x2^2 = 2*x0 => x2 = -sqrt(2*x0).  Minimizing
    f(x0) = x0 - 3*sqrt(2*x0) gives x0 = 9/2, x2 = -3, obj = -9/2.

    KKT (lambda on g = x2^2 + x3^2 - x0*x1 <= 0):
      x0: 1 - lambda*x1 = 0  =>  lambda = 1/2
      x2: 3 + lambda*2*x2 = 0
      EQ x1: DualValue = -lambda*x0 = -9/4;  EQ x3: 0.

    At the optimum the lifted slacks are s0 = h*(x0+x1) > 0 and s1 = h*(x0-x1) > 0
    with h = 1/2, so recovery must use the rotated path with nonzero s1.
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
    Min x0 + x1  s.t. x0=0, x1=0, x2^2+x3^2 <= x0*x1, x0,x1>=0  =>  apex of lifted cone.

    Both rotated heads are pinned to zero, so s0=s1=0 at the optimum.  grad g = 0 and the QC
    multiplier is not unique; recovery exports mu = 0 by convention.
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
    General PSD lift: 2*x0^2 + 2*x0*x1 + 2*x1^2 - 1 <= 0, x0 - x1 = 0.
    Optimum x0=x1=-1/sqrt(6), obj=-2/sqrt(6), lambda=1/sqrt(6).
    """
    problem = Problem("path_general")
    x0 = problem.addVariable(lb=-np.inf, name="x0")
    x1 = problem.addVariable(lb=-np.inf, name="x1")
    problem.setObjective(x0 + x1)
    problem.addConstraint(
        2 * x0 * x0 + 2 * x0 * x1 + 2 * x1 * x1 - 1 <= 0, name="general_qc"
    )
    problem.addConstraint(x0 - x1 == 0)
    return problem, (x0, x1)


def test_socp_translation_lorentz_path():
    problem, (_, _, t) = build_lorentz_path_problem()
    _solve_and_assert_translation_path(
        problem,
        "lorentz",
        expected_obj=1.0,
        var_expect=[(t, 1.0)],
    )


def test_socp_translation_lorentz_degenerate():
    """At the Lorentz cone apex, the QC dual is not unique; export mu = 0."""
    problem, (x1, x2, t) = build_lorentz_degenerate_problem()
    _solve_barrier(problem)
    assert x1.Value == pytest.approx(0.0, abs=PRIMAL_TOL)
    assert x2.Value == pytest.approx(0.0, abs=PRIMAL_TOL)
    assert t.Value == pytest.approx(0.0, abs=PRIMAL_TOL)
    assert problem.ObjValue == pytest.approx(0.0, abs=OBJ_TOL)
    qc = next(c for c in problem.getConstraints() if c.is_quadratic)
    assert qc.DualValue == pytest.approx(0.0, abs=DUAL_TOL)


def test_socp_translation_affine_path():
    problem, (x0, x2, _) = build_affine_path_problem()
    _solve_and_assert_translation_path(
        problem,
        "affine",
        expected_obj=-1.0,
        var_expect=[(x2, -1.0), (x0, 1.0)],
    )


def test_socp_translation_rotated_degenerate():
    """At the rotated lift apex (s0=s1=0), the QC dual is not unique; export mu = 0."""
    problem, (x0, x1, x2, x3) = build_rotated_degenerate_problem()
    _solve_barrier(problem)
    assert x0.Value == pytest.approx(0.0, abs=PRIMAL_TOL)
    assert x1.Value == pytest.approx(0.0, abs=PRIMAL_TOL)
    assert x2.Value == pytest.approx(0.0, abs=PRIMAL_TOL)
    assert x3.Value == pytest.approx(0.0, abs=PRIMAL_TOL)
    assert problem.ObjValue == pytest.approx(0.0, abs=OBJ_TOL)
    qc = next(c for c in problem.getConstraints() if c.is_quadratic)
    assert qc.DualValue == pytest.approx(0.0, abs=DUAL_TOL)


def test_socp_translation_rotated_path():
    problem, (x0, x1, x2, x3) = build_rotated_path_problem()
    _solve_and_assert_translation_path(
        problem,
        "rotated",
        expected_obj=-4.5,
        var_expect=[
            (x0, 4.5),
            (x1, 2.0),
            (x2, -3.0),
            (x3, 0.0),
        ],
    )


def test_socp_translation_general_path():
    problem, (x0, x1) = build_general_path_problem()
    expected_x = -1.0 / _SQRT6
    _solve_and_assert_translation_path(
        problem,
        "general",
        expected_obj=-2.0 / _SQRT6,
        var_expect=[(x0, expected_x), (x1, expected_x)],
    )


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
