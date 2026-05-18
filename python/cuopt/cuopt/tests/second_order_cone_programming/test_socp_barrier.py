# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for SOCP problems solved via the barrier method.

Quadratic constraints (QCMATRIX form) are converted to second-order cones and
solved with the barrier interior-point method.
"""

import numpy as np
import pytest

from cuopt.linear_programming import data_model, solver, solver_settings
from cuopt.linear_programming.problem import Problem
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_BARRIER_RELATIVE_COMPLEMENTARITY_TOLERANCE,
    CUOPT_BARRIER_RELATIVE_FEASIBILITY_TOLERANCE,
    CUOPT_BARRIER_RELATIVE_OPTIMALITY_TOLERANCE,
    CUOPT_METHOD,
    CUOPT_PRESOLVE,
)
from cuopt.linear_programming.solver_settings import SolverMethod


def _barrier_settings():
    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)
    settings.set_parameter(CUOPT_PRESOLVE, 0)
    return settings


def _solve_socp(dm, settings=None):
    if settings is None:
        settings = _barrier_settings()
    return solver.Solve(dm, settings)


def _lorentz_qcoo(variable_indices):
    """COO triplets for sum_{i>0} x_i^2 - x_0^2 <= 0 on variable_indices."""
    indices = [int(i) for i in variable_indices]
    q_values = [-1.0] + [1.0] * (len(indices) - 1)
    return (
        np.array(q_values, dtype=np.float64),
        np.array(indices, dtype=np.int32),
        np.array(indices, dtype=np.int32),
    )


def _build_socp_min_x0_model():
    """min x0 s.t. x1=1 and Lorentz cone on (x0,x1,x2); optimal obj=1."""
    dm = data_model.DataModel()

    A_values = np.array([1.0], dtype=np.float64)
    A_indices = np.array([1], dtype=np.int32)
    A_offsets = np.array([0, 1], dtype=np.int32)
    dm.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    dm.set_constraint_lower_bounds(np.array([1.0], dtype=np.float64))
    dm.set_constraint_upper_bounds(np.array([1.0], dtype=np.float64))

    dm.set_objective_coefficients(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    dm.set_variable_lower_bounds(np.array([0.0, 0.0, 0.0], dtype=np.float64))
    dm.set_variable_upper_bounds(
        np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    )

    qv, qr, qc = _lorentz_qcoo([0, 1, 2])
    dm.add_quadratic_constraint(
        constraint_row_index=1,
        constraint_row_name="soc",
        quadratic_values=qv,
        quadratic_row_indices=qr,
        quadratic_col_indices=qc,
    )
    return dm


def _rotated_qcoo(variable_indices):
    """COO triplets for -2 x_0 x_1 + sum_{i>1} x_i^2 <= 0 on variable_indices."""
    indices = [int(i) for i in variable_indices]
    head0, head1 = indices[0], indices[1]
    tails = indices[2:]
    q_rows = [head0, head1] + tails
    q_cols = [head1, head0] + tails
    q_values = [-1.0, -1.0] + [1.0] * len(tails)
    return (
        np.array(q_values, dtype=np.float64),
        np.array(q_rows, dtype=np.int32),
        np.array(q_cols, dtype=np.int32),
    )


def test_quadratic_constraint_api():
    dm = data_model.DataModel()

    dm.add_quadratic_constraint(
        constraint_row_index=0,
        constraint_row_name="qc0",
        quadratic_values=np.array([-1.0, 1.0, 1.0], dtype=np.float64),
        quadratic_row_indices=np.array([0, 1, 2], dtype=np.int32),
        quadratic_col_indices=np.array([0, 1, 2], dtype=np.int32),
    )
    dm.add_quadratic_constraint(
        constraint_row_index=1,
        constraint_row_name="qc_ge",
        sense="G",
        quadratic_values=np.array([1.0, -1.0, -1.0], dtype=np.float64),
        quadratic_row_indices=np.array([0, 1, 2], dtype=np.int32),
        quadratic_col_indices=np.array([0, 1, 2], dtype=np.int32),
    )
    qcs = dm.get_quadratic_constraints()
    assert qcs[1]["constraint_row_type"] == "G"
    assert qcs[1]["quadratic_values"][0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Equality quadratic"):
        dm.add_quadratic_constraint(
            constraint_row_index=2,
            sense="E",
            quadratic_values=np.array([-1.0], dtype=np.float64),
            quadratic_row_indices=np.array([0], dtype=np.int32),
            quadratic_col_indices=np.array([0], dtype=np.int32),
        )
    dm.clear_quadratic_constraints()

def test_quadratic_constraint_problem_solve():
    """Problem.addQuadraticConstraint + solve (Lorentz via expression)."""
    prob = Problem()
    x0 = prob.addVariable(lb=0.0, name="x0")
    x1 = prob.addVariable(lb=0.0, name="x1")
    x2 = prob.addVariable(lb=0.0, name="x2")
    prob.addConstraint(x1 == 1.0)
    prob.addQuadraticConstraint(-x0 * x0 + x1 * x1 + x2 * x2 <= 0, name="soc")
    prob.setObjective(x0)

    solution = prob.solve(_barrier_settings())

    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert len(prob.getQuadraticConstraints()) == 1


def test_quadratic_constraint_problem_solve_ge_sense():
    """>= quadratic rows are converted to <= form when building SOCP for the barrier."""
    prob = Problem()
    x0 = prob.addVariable(lb=0.0, name="x0")
    x1 = prob.addVariable(lb=0.0, name="x1")
    x2 = prob.addVariable(lb=0.0, name="x2")
    prob.addConstraint(x1 == 1.0)
    prob.addQuadraticConstraint(x0 * x0 - x1 * x1 - x2 * x2 >= 0, name="soc")
    prob.setObjective(x0)

    solution = prob.solve(_barrier_settings())

    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert prob.model.get_quadratic_constraints()[0]["constraint_row_type"] == "G"


def test_rotated_second_order_cone_min_tail():
    # minimize x2
    # subject to x0 = 1, x1 = 1
    #            2 x0 x1 >= x2^2   (rotated SOC on (x0, x1, x2))
    #
    # Optimal: x* = (1, 1, 0), obj* = 0

    dm = data_model.DataModel()

    A_values = np.array([1.0, 1.0], dtype=np.float64)
    A_indices = np.array([0, 1], dtype=np.int32)
    A_offsets = np.array([0, 1, 2], dtype=np.int32)
    dm.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    dm.set_constraint_lower_bounds(np.array([1.0, 1.0], dtype=np.float64))
    dm.set_constraint_upper_bounds(np.array([1.0, 1.0], dtype=np.float64))

    dm.set_objective_coefficients(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    dm.set_variable_lower_bounds(np.array([0.0, 0.0, 0.0], dtype=np.float64))
    dm.set_variable_upper_bounds(
        np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    )

    qv, qr, qc = _rotated_qcoo([0, 1, 2])
    dm.add_quadratic_constraint(
        constraint_row_index=2,
        constraint_row_name="rsoc",
        quadratic_values=qv,
        quadratic_row_indices=qr,
        quadratic_col_indices=qc,
    )

    solution = _solve_socp(dm)

    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(0.0, abs=1e-3)
    x = solution.get_primal_solution()
    assert x[0] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert x[1] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert abs(x[2]) == pytest.approx(0.0, abs=1e-3)


def test_multiple_rotated_second_order_cone_constraints():
    # Two disjoint rotated cones:
    #   2 x0 x1 >= x2^2  with x0 = x1 = 1
    #   2 x3 x4 >= x5^2  with x3 = 2, x4 = 1
    # minimize x2 + x5  -> 0

    dm = data_model.DataModel()

    A_values = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    A_indices = np.array([0, 1, 3, 4], dtype=np.int32)
    A_offsets = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    dm.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    dm.set_constraint_lower_bounds(np.array([1.0, 1.0, 2.0, 1.0], dtype=np.float64))
    dm.set_constraint_upper_bounds(np.array([1.0, 1.0, 2.0, 1.0], dtype=np.float64))

    dm.set_objective_coefficients(
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )
    dm.set_variable_lower_bounds(np.zeros(6, dtype=np.float64))
    dm.set_variable_upper_bounds(np.full(6, np.inf, dtype=np.float64))

    qv0, qr0, qc0 = _rotated_qcoo([0, 1, 2])
    dm.add_quadratic_constraint(
        constraint_row_index=4,
        constraint_row_name="rsoc1",
        quadratic_values=qv0,
        quadratic_row_indices=qr0,
        quadratic_col_indices=qc0,
    )
    qv1, qr1, qc1 = _rotated_qcoo([3, 4, 5])
    dm.add_quadratic_constraint(
        constraint_row_index=5,
        constraint_row_name="rsoc2",
        quadratic_values=qv1,
        quadratic_row_indices=qr1,
        quadratic_col_indices=qc1,
    )

    solution = _solve_socp(dm)

    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(0.0, abs=1e-3)
    x = solution.get_primal_solution()
    assert x[0] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert x[1] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert abs(x[2]) == pytest.approx(0.0, abs=1e-3)
    assert x[3] == pytest.approx(2.0, rel=1e-3, abs=1e-3)
    assert x[4] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert abs(x[5]) == pytest.approx(0.0, abs=1e-3)


def test_multiple_quadratic_constraints():
    """Port of C++ mixed_linear_and_two_soc_blocks (solve_barrier.cu).

    Two Lorentz QCMATRIX rows via add_quadratic_constraint.
    Variables [l1, l2 | t1, u1, v1 | t2, u2, v2]; Lorentz cones on (t1,u1,v1) and (t2,u2,v2).

    minimize   t1 + t2
    subject to l1 - u1 = 0, l2 - u2 = 0, l1 + l2 = 3, l1 - l2 = 1

    Optimal: l1*=2, l2*=1, t1*=2, u1*=2, v1*=0, t2*=1, u2*=1, v2*=0, obj*=3.
    """
    dm = data_model.DataModel()

    A_values = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64)
    A_indices = np.array([0, 3, 1, 6, 0, 1, 0, 1], dtype=np.int32)
    A_offsets = np.array([0, 2, 4, 6, 8], dtype=np.int32)
    dm.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    dm.set_constraint_lower_bounds(np.array([0.0, 0.0, 3.0, 1.0], dtype=np.float64))
    dm.set_constraint_upper_bounds(np.array([0.0, 0.0, 3.0, 1.0], dtype=np.float64))

    dm.set_objective_coefficients(
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    )
    dm.set_variable_lower_bounds(np.zeros(8, dtype=np.float64))
    dm.set_variable_upper_bounds(np.full(8, np.inf, dtype=np.float64))

    qv0, qr0, qc0 = _lorentz_qcoo([2, 3, 4])
    dm.add_quadratic_constraint(
        constraint_row_index=4,
        constraint_row_name="soc1",
        quadratic_values=qv0,
        quadratic_row_indices=qr0,
        quadratic_col_indices=qc0,
    )

    qv1, qr1, qc1 = _lorentz_qcoo([5, 6, 7])
    dm.add_quadratic_constraint(
        constraint_row_index=5,
        constraint_row_name="soc2",
        quadratic_values=qv1,
        quadratic_row_indices=qr1,
        quadratic_col_indices=qc1,
    )

    solution = _solve_socp(dm)

    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(3.0, rel=1e-3, abs=1e-3)
    x = solution.get_primal_solution()
    assert x[0] == pytest.approx(2.0, rel=1e-3, abs=1e-3)
    assert x[1] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert x[2] == pytest.approx(2.0, rel=1e-3, abs=1e-3)
    assert x[3] == pytest.approx(2.0, rel=1e-3, abs=1e-3)
    assert abs(x[4]) == pytest.approx(0.0, abs=1e-3)
    assert x[5] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert x[6] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert abs(x[7]) == pytest.approx(0.0, abs=1e-3)


def test_socp_min_x0_norm_constraint():
    # minimize x0
    # subject to x1 = 1
    #            (x0, x1, x2) in Lorentz cone  (x0^2 >= x1^2 + x2^2, x0 >= 0)
    #
    # Optimal: x* = (1, 1, 0), obj* = 1

    solution = _solve_socp(_build_socp_min_x0_model())

    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    x = solution.get_primal_solution()
    assert x[0] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert x[1] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert abs(x[2]) == pytest.approx(0.0, abs=1e-3)


def test_socp_mixed_linear_and_cone_block():
    # Variables [l, t, u, v]; Lorentz cone on (t, u, v).
    #
    # minimize l
    # subject to l - t = 0
    #            u = 1
    #            (t, u, v) in Lorentz cone
    #
    # Optimal: l* = 1, t* = 1, u* = 1, v* = 0, obj* = 1.

    dm = data_model.DataModel()

    A_values = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    A_indices = np.array([0, 1, 2], dtype=np.int32)
    A_offsets = np.array([0, 2, 3], dtype=np.int32)
    dm.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    dm.set_constraint_lower_bounds(np.array([0.0, 1.0], dtype=np.float64))
    dm.set_constraint_upper_bounds(np.array([0.0, 1.0], dtype=np.float64))

    dm.set_objective_coefficients(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    dm.set_variable_lower_bounds(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64))
    dm.set_variable_upper_bounds(
        np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
    )

    qv, qr, qc = _lorentz_qcoo([1, 2, 3])
    dm.add_quadratic_constraint(
        constraint_row_index=2,
        constraint_row_name="soc",
        quadratic_values=qv,
        quadratic_row_indices=qr,
        quadratic_col_indices=qc,
    )

    solution = _solve_socp(dm)

    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    x = solution.get_primal_solution()
    assert x[0] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert x[1] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert x[2] == pytest.approx(1.0, rel=1e-3, abs=1e-3)
    assert abs(x[3]) == pytest.approx(0.0, abs=1e-3)


def test_socp_barrier_relative_tolerances():
    settings = _barrier_settings()
    settings.set_parameter(CUOPT_BARRIER_RELATIVE_FEASIBILITY_TOLERANCE, 1e-10)
    settings.set_parameter(CUOPT_BARRIER_RELATIVE_OPTIMALITY_TOLERANCE, 1e-10)
    settings.set_parameter(CUOPT_BARRIER_RELATIVE_COMPLEMENTARITY_TOLERANCE, 1e-10)

    solution = solver.Solve(_build_socp_min_x0_model(), settings)

    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(1.0, rel=1e-4, abs=1e-4)
