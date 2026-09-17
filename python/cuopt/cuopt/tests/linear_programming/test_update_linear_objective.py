# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QP barrier ``update_linear_objective`` vs a full Solve on a fresh DataModel."""

import numpy as np
import pytest

from cuopt.linear_programming import data_model, solver
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_BARRIER_PRESOLVE_BOUND_FREE_VARIABLES,
    CUOPT_METHOD,
    CUOPT_SEQUENCE_SOLVE,
)
from cuopt.linear_programming.solver_settings import (
    SolverMethod,
    SolverSettings,
)

# Same QP as cpp/tests/qp/unit_tests/two_variable_test.cu:
#   min  x1^2 + 4 x2^2 + c1 x1 + c2 x2
#   s.t. x1 + x2 >= 5,  0 <= x_i <= 10
# Unconstrained min at x1 = -c1/2, x2 = -c2/8.
C0 = np.array([-8.0, -16.0])
C1 = np.array([-10.0, -16.0])
OBJ_TOL = 1e-5
_CACHE_REUSE = "Barrier: reusing cache (skip convert/presolve/scaling)"


def _qp(c, *, maximize=False, q_diagonal=(1.0, 4.0), lower=(0.0, 0.0)):
    dm = data_model.DataModel()
    dm.set_csr_constraint_matrix(
        np.array([1.0, 1.0]),
        np.array([0, 1], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
    )
    dm.set_constraint_bounds(np.array([5.0], dtype=np.float64))
    dm.set_row_types(np.array(["G"]))
    dm.set_objective_coefficients(np.asarray(c, dtype=np.float64))
    dm.set_quadratic_objective_matrix(
        np.asarray(q_diagonal, dtype=np.float64),
        np.array([0, 1], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.int32),
    )
    dm.set_variable_lower_bounds(np.asarray(lower, dtype=np.float64))
    dm.set_variable_upper_bounds(np.array([10.0, 10.0]))
    dm.set_maximize(maximize)
    return dm


def _barrier_settings(*, sequence_solve):
    settings = SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)
    settings.set_parameter(CUOPT_BARRIER_PRESOLVE_BOUND_FREE_VARIABLES, 0)
    settings.set_parameter(CUOPT_SEQUENCE_SOLVE, sequence_solve)
    return settings


def test_update_linear_objective_matches_full_solve(capfd):
    dm = _qp(C0)
    settings = _barrier_settings(sequence_solve=True)

    sol0 = solver.Solve(dm, settings)
    log0 = capfd.readouterr()
    assert sol0.get_termination_reason() == "Optimal"
    assert sol0.get_primal_objective() == pytest.approx(-32.0, abs=OBJ_TOL)
    assert _CACHE_REUSE not in log0.out + log0.err

    dm.update_linear_objective(C1)
    sol_reuse = solver.Solve(dm, settings)
    log_reuse = capfd.readouterr()
    assert sol_reuse.get_termination_reason() == "Optimal"
    assert _CACHE_REUSE in log_reuse.out + log_reuse.err

    sol_full = solver.Solve(_qp(C1), _barrier_settings(sequence_solve=False))
    assert sol_full.get_termination_reason() == "Optimal"
    assert sol_reuse.get_primal_objective() == pytest.approx(
        sol_full.get_primal_objective(), abs=OBJ_TOL, rel=1e-8
    )
    assert sol_reuse.get_primal_objective() == pytest.approx(
        -41.0, abs=OBJ_TOL
    )


def test_update_linear_objective_honors_maximize_and_translated_bounds(capfd):
    # max -x1^2 - 4 x2^2 + c1 x1 + c2 x2,  3 <= x_i <= 10
    q_diagonal = (-1.0, -4.0)
    lower = (3.0, 4.0)
    dm = _qp((8.0, 16.0), maximize=True, q_diagonal=q_diagonal, lower=lower)
    settings = _barrier_settings(sequence_solve=True)

    sol0 = solver.Solve(dm, settings)
    capfd.readouterr()
    assert sol0.get_termination_reason() == "Optimal"
    assert sol0.get_primal_objective() == pytest.approx(16.0, abs=OBJ_TOL)

    new_objective = np.array([10.0, 16.0])
    dm.update_linear_objective(new_objective)
    sol_reuse = solver.Solve(dm, settings)
    log_reuse = capfd.readouterr()
    assert sol_reuse.get_termination_reason() == "Optimal"
    assert _CACHE_REUSE in log_reuse.out + log_reuse.err

    sol_full = solver.Solve(
        _qp(
            new_objective,
            maximize=True,
            q_diagonal=q_diagonal,
            lower=lower,
        ),
        _barrier_settings(sequence_solve=False),
    )
    assert sol_full.get_termination_reason() == "Optimal"
    assert sol_reuse.get_primal_objective() == pytest.approx(
        sol_full.get_primal_objective(), abs=OBJ_TOL, rel=1e-8
    )
    assert sol_reuse.get_primal_objective() == pytest.approx(25.0, abs=OBJ_TOL)
