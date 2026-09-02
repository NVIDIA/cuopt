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

from cuopt.linear_programming import Read
from cuopt.linear_programming.problem import EQ, GE, LE, MAXIMIZE, Problem
from cuopt.linear_programming.solver.solver_parameters import CUOPT_METHOD
from cuopt.linear_programming.solver_settings import (
    SolverMethod,
    SolverSettings,
)

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


def build_rotated_soc_natural_cross_term_example() -> tuple[Problem, tuple]:
    """Rotated SOC with the natural single cross term ``-2*t*u``.

    max  v0 + v1
    s.t. t = 0.5, u = 1.0
         t + u + v0 + v1 <= 100
         v0^2 + v1^2 - 2*t*u <= 0
    """
    problem = Problem("rotated_soc_natural_cross")
    t = problem.addVariable(lb=0.5, ub=0.5, name="t")
    u = problem.addVariable(lb=1.0, ub=1.0, name="u")
    v0 = problem.addVariable(lb=-np.inf, name="v0")
    v1 = problem.addVariable(lb=-np.inf, name="v1")
    problem.addConstraint(t + u + v0 + v1 <= 100.0, name="slack")
    problem.addConstraint(
        v0 * v0 + v1 * v1 - 2 * t * u <= 0.0, name="rotated_soc"
    )
    problem.setObjective(v0 + v1, sense=MAXIMIZE)
    return problem, (t, u, v0, v1)


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
        # Classical slack/surplus from populate_solution (non-negative if feasible).
        slack = constr.Slack
        if constr.Sense in (LE, GE):
            assert slack >= -FEAS_TOL
        else:
            assert constr.Sense == EQ
            assert slack == pytest.approx(0.0, abs=FEAS_TOL)


def _solve(problem: Problem):
    solution = problem.solve(_barrier_settings())
    assert problem.Status.name == "Optimal"
    return solution


def test_socp_3_barrier_solution():
    problem, (x0, x1, h1, h2, h3) = build_socp_3()
    solution = _solve(problem)
    _assert_solution_on_original_model(problem, solution)
    _assert_feasible(problem)

    expected_obj = -1.932105
    expected_x = (0.83666003, -0.54772256)
    assert problem.ObjValue == pytest.approx(expected_obj, abs=OBJ_TOL)
    assert x0.Value == pytest.approx(expected_x[0], abs=PRIMAL_TOL)
    assert x1.Value == pytest.approx(expected_x[1], abs=PRIMAL_TOL)
    assert h1.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
    assert h2.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
    assert h3.Value == pytest.approx(1.0, abs=PRIMAL_TOL)


def test_rotated_soc_natural_cross_term_barrier_solution():
    """Barrier solve for rotated SOC with a single ``-2*t*u`` cross term."""
    problem, (t, u, v0, v1) = build_rotated_soc_natural_cross_term_example()
    solution = _solve(problem)
    _assert_solution_on_original_model(problem, solution)
    _assert_feasible(problem)

    # Single canonical cross term -2*t*u; optimum max v0+v1 is sqrt(2).
    sqrt2 = np.sqrt(2.0)
    expected_obj = sqrt2
    expected_v = sqrt2 / 2.0
    assert problem.ObjValue == pytest.approx(expected_obj, abs=OBJ_TOL)
    assert t.Value == pytest.approx(0.5, abs=PRIMAL_TOL)
    assert u.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
    assert v0.Value == pytest.approx(expected_v, abs=PRIMAL_TOL)
    assert v1.Value == pytest.approx(expected_v, abs=PRIMAL_TOL)


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
    from cuopt.linear_programming.problem import MINIMIZE

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


# Same model as test_maximize_with_quadratic_constraint written as MPS (as a
# minimization). QC0 is the binding row: without it the optimum is -10.
QC_MPS = """NAME          QCREAD
ROWS
 N  OBJ
 L  LIN0
 L  QC0
COLUMNS
    x         OBJ              -1
    x         LIN0              1
    y         OBJ              -1
    y         LIN0              1
RHS
    RHS1      LIN0             10
    RHS1      QC0               6
QCMATRIX   QC0
    x         x                 2
    x         y                 1
    y         x                 1
    y         y                 2
BOUNDS
 MI BND       x
 MI BND       y
ENDATA
"""


def _write_qc_mps(tmp_path) -> str:
    path = tmp_path / "qc_read.mps"
    path.write_text(QC_MPS)
    return str(path)


def test_read_keeps_quadratic_constraints(tmp_path):
    """QCMATRIX rows parsed by read() must reach the solver."""
    path = _write_qc_mps(tmp_path)
    problem = Problem.read(path)

    assert problem.NumConstraints == 2
    quad = problem.getQuadraticConstraints()
    assert len(quad) == 1

    # Each row must mirror the bundle the reader parsed, in whatever form the
    # reader normalized it to.
    bundle = Read(path).get_quadratic_constraints()[0]
    assert quad[0].ConstraintName == bundle["constraint_row_name"]
    assert quad[0].Sense == bundle["constraint_row_type"]
    assert quad[0].rhs_value == pytest.approx(bundle["rhs_value"])
    for field in ("rows", "cols", "vals", "linear_indices", "linear_values"):
        np.testing.assert_allclose(getattr(quad[0], field), bundle[field])

    # The file DataModel is the model to solve; rebuilding it from Python is
    # what used to drop the quadratic rows.
    model = problem.model
    solution = _solve(problem)
    assert problem.model is model

    _assert_solution_on_original_model(problem, solution)
    _assert_feasible(problem)

    x, y = problem.getVariables()
    assert problem.ObjValue == pytest.approx(-2.0, abs=OBJ_TOL)
    assert x.Value == pytest.approx(1.0, abs=PRIMAL_TOL)
    assert y.Value == pytest.approx(1.0, abs=PRIMAL_TOL)


def test_read_then_rebuild_keeps_quadratic_constraints(tmp_path):
    """A rebuild after read() must re-emit the quadratic rows."""
    problem = Problem.read(_write_qc_mps(tmp_path))
    # update() drops the cached CSR, so writeMPS rebuilds the DataModel from
    # the Python objects instead of reusing the one that was read.
    problem.getConstraint(0).RHS = 9.0
    problem.update()

    round_trip = tmp_path / "round_trip.mps"
    problem.writeMPS(str(round_trip))
    assert "QCMATRIX" in round_trip.read_text()

    reread = Problem.read(str(round_trip))
    assert len(reread.getQuadraticConstraints()) == 1
    _solve(reread)
    _assert_feasible(reread)
    assert reread.ObjValue == pytest.approx(-2.0, abs=OBJ_TOL)
