# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Barrier cache reuse (``sequence_solve``) for ``DataModel.update_rhs``.

Every re-solve through the cache is compared against a fresh full solve of the
same model, so no assertion depends on a hand-derived optimum. The models are
picked to exercise the parts of the RHS crush that a one-row QP leaves as
no-ops: mixed row senses, presolve-dropped empty rows, non-unit row scaling,
and the shift that comes from translating nonzero variable lower bounds.

Each test also asserts the reuse path actually ran. Without that, a test passes
just as happily when the gate rejects the model and the solver quietly falls
back to a full solve, which returns the same answer.
"""

import numpy as np
import pytest

from cuopt.linear_programming import (
    data_model,
    solver,
    solver_settings,
)

REUSE_LOG = "reusing cache"
RHS_INFEASIBLE_LOG = "update_rhs made an empty constraint row infeasible"
IPM_LOG = "Optimal solution found"


def _sequence_settings():
    settings = solver_settings.SolverSettings()
    settings.sequence_solve = True
    # The reuse gate requires exactly 0. The default of -1 (automatic) leaves
    # every solve on the full path, silently, with correct results.
    settings.set_parameter("barrier_presolve_bound_free_variables", 0)
    return settings


def _build(values, indices, offsets, rhs, senses, lower, upper):
    """A QP with objective x^T x, so the model is always barrier-eligible."""
    n = len(lower)
    model = data_model.DataModel()
    model.set_csr_constraint_matrix(
        np.asarray(values, dtype=np.float64),
        np.asarray(indices, dtype=np.int32),
        np.asarray(offsets, dtype=np.int32),
    )
    model.set_constraint_bounds(np.asarray(rhs, dtype=np.float64))
    model.set_row_types(senses)
    model.set_objective_coefficients(np.zeros(n))
    model.set_quadratic_objective_matrix(
        np.ones(n),
        np.arange(n, dtype=np.int32),
        np.arange(n + 1, dtype=np.int32),
    )
    model.set_variable_lower_bounds(np.asarray(lower, dtype=np.float64))
    model.set_variable_upper_bounds(np.asarray(upper, dtype=np.float64))
    return model


def _solve(model, settings, capfd):
    """Solve and return the log this solve alone produced."""
    capfd.readouterr()
    solution = solver.Solve(model, settings)
    return solution, capfd.readouterr().out


def _full_solve(model_args, rhs):
    """Oracle: a fresh model with default settings, no cache in play."""
    args = dict(model_args, rhs=rhs)
    return solver.Solve(_build(**args), solver_settings.SolverSettings())


def _assert_matches_oracle(reused, oracle):
    assert reused.get_termination_reason() == "Optimal"
    assert reused.get_termination_reason() == oracle.get_termination_reason()
    assert reused.get_primal_objective() == pytest.approx(
        oracle.get_primal_objective(), rel=1e-6, abs=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(reused.get_primal_solution()),
        np.asarray(oracle.get_primal_solution()),
        rtol=1e-5,
        atol=1e-5,
    )


# x0 + x1 == b0 ; x1 + x2 <= b1 ; x0 + x2 >= b2. The equality and the greater
# than row are both active at the optimum, so the 'E' and 'G' crush paths
# (the latter negates the row) both affect the answer.
MIXED_SENSES = dict(
    values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    indices=[0, 1, 1, 2, 0, 2],
    offsets=[0, 2, 4, 6],
    senses="ELG",
    lower=[0.0, 0.0, 0.0],
    upper=[10.0, 10.0, 10.0],
)

# Row norms differ by seven orders of magnitude, so equilibration cannot leave
# row_scales at 1 and the crush has to divide the new RHS by the right ones.
BADLY_SCALED = dict(
    values=[1e4, 1e4, 1e-3, -1e-3],
    indices=[0, 1, 0, 1],
    offsets=[0, 2, 4],
    senses="GL",
    lower=[0.0, 0.0],
    upper=[10.0, 10.0],
)

# Nonzero variable lower bounds make presolve translate x = x' + l, which
# subtracts sum_j a_ij * l_j from each row's RHS. That constant is recorded
# once as rhs_shift and re-added on every update. Here it is -7 on the first
# row, so dropping it would solve x0 + x1 >= b0 + 7 and land far from the
# oracle rather than merely a tolerance away from it.
LOWER_BOUNDED = dict(
    values=[1.0, 1.0, 1.0, -1.0],
    indices=[0, 1, 0, 1],
    offsets=[0, 2, 4],
    senses="GL",
    lower=[3.0, 4.0],
    upper=[20.0, 20.0],
)

# Row 0 has no coefficients, so presolve drops it and it never reaches the
# barrier problem. Its RHS can only be checked for feasibility, not applied.
EMPTY_ROW = dict(
    values=[1.0, 1.0],
    indices=[0, 1],
    offsets=[0, 0, 2],
    senses="EG",
    lower=[0.0, 0.0],
    upper=[10.0, 10.0],
)


@pytest.mark.parametrize(
    "model_args,first_rhs,updates",
    [
        (MIXED_SENSES, [5.0, 8.0, 3.0], [[6.0, 7.0, 4.0], [4.0, 9.0, 2.5]]),
        (BADLY_SCALED, [2e4, 5e-3], [[3e4, 2e-3], [1e4, 8e-3]]),
        (LOWER_BOUNDED, [12.0, 1.0], [[15.0, 2.0], [9.0, 0.5]]),
    ],
    ids=["mixed_senses", "row_scaling", "rhs_shift"],
)
def test_update_rhs_matches_full_solve(model_args, first_rhs, updates, capfd):
    """Reused solves must agree with a fresh full solve of the same model."""
    settings = _sequence_settings()
    model = _build(**dict(model_args, rhs=first_rhs))

    first, _ = _solve(model, settings, capfd)
    assert first.get_termination_reason() == "Optimal"

    for rhs in updates:
        model.update_rhs(np.asarray(rhs, dtype=np.float64))
        reused, log = _solve(model, settings, capfd)
        assert REUSE_LOG in log, "update_rhs fell back to a full solve"
        _assert_matches_oracle(reused, _full_solve(model_args, rhs))


def test_update_rhs_keeps_dropped_empty_row(capfd):
    """An empty row still satisfied by the new RHS must not block reuse."""
    settings = _sequence_settings()
    model = _build(**dict(EMPTY_ROW, rhs=[0.0, 2.0]))

    first, log = _solve(model, settings, capfd)
    assert first.get_termination_reason() == "Optimal"
    assert "empty rows" in log, "presolve did not drop the empty row"

    rhs = [0.0, 4.0]
    model.update_rhs(np.asarray(rhs, dtype=np.float64))
    reused, log = _solve(model, settings, capfd)
    assert REUSE_LOG in log
    _assert_matches_oracle(reused, _full_solve(EMPTY_ROW, rhs))


def test_update_rhs_infeasible_empty_row_short_circuits(capfd):
    """A dropped 'E' row needs 0 == b_i; violating it is infeasible.

    The row has no variables, so it is either satisfied for every x or for
    none. That makes the verdict exact and lets the next Solve answer without
    running IPM. The cache is kept, so a later feasible RHS still reuses it.
    """
    settings = _sequence_settings()
    model = _build(**dict(EMPTY_ROW, rhs=[0.0, 2.0]))
    assert _solve(model, settings, capfd)[0].get_termination_reason() == (
        "Optimal"
    )

    model.update_rhs(np.array([1.0, 4.0]))
    infeasible, log = _solve(model, settings, capfd)
    assert infeasible.get_termination_reason() == "PrimalInfeasible"
    assert RHS_INFEASIBLE_LOG in log
    assert IPM_LOG not in log, "IPM ran despite a provably infeasible row"

    # Same cache, feasible RHS again.
    rhs = [0.0, 4.0]
    model.update_rhs(np.asarray(rhs, dtype=np.float64))
    recovered, log = _solve(model, settings, capfd)
    assert REUSE_LOG in log, "cache was discarded by the infeasible update"
    _assert_matches_oracle(recovered, _full_solve(EMPTY_ROW, rhs))


def test_update_rhs_rejects_wrong_length():
    """Length is validated against the cached user row count."""
    settings = _sequence_settings()
    model = _build(**dict(MIXED_SENSES, rhs=[5.0, 8.0, 3.0]))
    assert solver.Solve(model, settings).get_termination_reason() == "Optimal"

    with pytest.raises(Exception, match="match the cached user row count"):
        model.update_rhs(np.array([1.0, 2.0]))
