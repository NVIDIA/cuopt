# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Barrier cache reuse (``sequence_solve``) for the DataModel update APIs.

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
    settings.set_parameter("sequence_solve", True)
    # barrier_presolve_bound_free_variables stays at its -1 default on purpose:
    # sequence_solve resolves automatic to 0, which the reuse assertions cover.
    return settings


def _build(
    values, indices, offsets, rhs, senses, lower, upper, objective=None
):
    """A QP with quadratic term x^T x, so the model is barrier-eligible."""
    n = len(lower)
    model = data_model.DataModel()
    model.set_csr_constraint_matrix(
        np.asarray(values, dtype=np.float64),
        np.asarray(indices, dtype=np.int32),
        np.asarray(offsets, dtype=np.int32),
    )
    model.set_constraint_bounds(np.asarray(rhs, dtype=np.float64))
    model.set_row_types(senses)
    model.set_objective_coefficients(
        np.zeros(n)
        if objective is None
        else np.asarray(objective, dtype=np.float64)
    )
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


def _full_solve(model_args, rhs, **overrides):
    """Oracle: a fresh model with default settings, no cache in play."""
    args = dict(model_args, rhs=rhs, **overrides)
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

# Nonzero lower bounds make presolve translate x = x' + l, subtracting
# sum_j a_ij * l_j from each row's RHS. That is rhs_shift, -7 on the first row
# here, so dropping it would solve x0 + x1 >= b0 + 7 and miss the oracle.
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


# Lower bounds away from zero make presolve translate x = x' + l, which folds
# sum_j c_j * l_j into obj_constant. An RHS update must leave that alone.
TRANSLATED = dict(
    values=[1.0, 1.0],
    indices=[0, 1],
    offsets=[0, 2],
    senses="G",
    lower=[3.0, 4.0],
    upper=[20.0, 20.0],
)


def test_update_rhs_leaves_objective_constant_alone(capfd):
    """An RHS update must not disturb the objective constant.

    obj_constant depends on c and on the translated lower bounds, not on b, so
    the reused objective has to stay exact for a lower-bounded model.
    """
    settings = _sequence_settings()
    model = _build(**dict(TRANSLATED, rhs=[12.0], objective=[2.0, 0.0]))

    first, _ = _solve(model, settings, capfd)
    assert first.get_termination_reason() == "Optimal"

    rhs = [15.0]
    model.update_rhs(np.asarray(rhs, dtype=np.float64))
    reused, log = _solve(model, settings, capfd)
    assert REUSE_LOG in log
    _assert_matches_oracle(
        reused, _full_solve(TRANSLATED, rhs, objective=[2.0, 0.0])
    )


# x1 is free but the rows imply bounds on it, so presolve's free-variable
# bounding leaves state in presolve_info that the reuse path cannot replay.
FREE_VARIABLE = dict(
    values=[1.0, 1.0, 1.0, 1.0],
    indices=[0, 1, 0, 1],
    offsets=[0, 2, 4],
    senses="GL",
    lower=[0.0, -np.inf],
    upper=[10.0, np.inf],
)
FREE_VARIABLE_RHS = [4.0, 8.0]


@pytest.mark.parametrize(
    "first_solve_bfv,expect_reuse",
    [(None, True), (1, False)],
    ids=["automatic_reuses", "explicit_bounding_refuses"],
)
def test_bounded_free_variables_block_reuse(
    first_solve_bfv, expect_reuse, capfd
):
    """A cache is only reusable if its own presolve left free variables alone.

    The two cases share a model, so the refusal in the second can only come
    from the first solve having bounded a free variable. Without the pair, an
    assertion that reuse did not happen would pass for any unrelated reason.
    """
    settings = _sequence_settings()
    if first_solve_bfv is not None:
        settings.set_parameter(
            "barrier_presolve_bound_free_variables", first_solve_bfv
        )
    model = _build(
        **dict(FREE_VARIABLE, rhs=FREE_VARIABLE_RHS, objective=[1.0, 0.0])
    )
    first, _ = _solve(model, settings, capfd)
    assert first.get_termination_reason() == "Optimal"

    new_objective = [3.0, -2.0]
    model.update_linear_objective(np.asarray(new_objective, dtype=np.float64))
    # Whatever the first solve asked for, this one asks for 0, which is what
    # the gate used to key on all by itself.
    settings.set_parameter("barrier_presolve_bound_free_variables", 0)
    second, log = _solve(model, settings, capfd)
    assert (REUSE_LOG in log) == expect_reuse

    _assert_matches_oracle(
        second,
        _full_solve(FREE_VARIABLE, FREE_VARIABLE_RHS, objective=new_objective),
    )


# Quadratic constraints reach the barrier as second-order cones. The conversion appends rows and
# permutes columns, so an update has to be mapped from model coordinates into the expanded layout
# the cache holds, and updates stay model-sized: the appended rows belong to the conversion.
# The conversion also rejects cone variables that carry an explicit upper bound or a nonzero
# lower bound, so the builders below leave the bounds of anything a cone touches open.


def _build_lorentz(rhs, objective, cone_head_free=False):
    """``||(x1, x2)|| <= t`` plus two linear rows, no quadratic objective.

    Written as ``-t^2 + x1^2 + x2^2 <= 0``, which the conversion recognizes
    and lifts by permuting ``(t, x1, x2)`` into a cone block, so all three are
    conic variables. With ``cone_head_free``, ``t`` loses its lower bound and
    row 1 becomes the singleton ``t >= b1`` that proves the head nonnegative.
    """
    n = 3
    model = data_model.DataModel()
    # row 0: x1 + x2 >= b0 ; row 1: t <= b1, or t >= b1 when the head is free
    model.set_csr_constraint_matrix(
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        np.array([1, 2, 0], dtype=np.int32),
        np.array([0, 2, 3], dtype=np.int32),
    )
    model.set_constraint_bounds(np.asarray(rhs, dtype=np.float64))
    model.set_row_types("GG" if cone_head_free else "GL")
    model.set_objective_coefficients(np.asarray(objective, dtype=np.float64))
    lower = np.zeros(n)
    if cone_head_free:
        lower[0] = -np.inf
    model.set_variable_lower_bounds(lower)
    model.set_variable_upper_bounds(np.full(n, np.inf))
    model.add_quadratic_constraint(
        vals=np.array([-1.0, 1.0, 1.0]),
        rows=np.array([0, 1, 2], dtype=np.int32),
        cols=np.array([0, 1, 2], dtype=np.int32),
        rhs_value=0.0,
        sense="L",
    )
    return model


def _build_general_cone(rhs, objective):
    """``x1^2 + x2^2 + 2*x1 <= 8``, a shifted disk, plus two linear rows.

    The linear part and the nonzero RHS keep this off the recognized Lorentz
    pattern, so the conversion takes its general path: it factors Q, adds cone
    variables of its own, and appends four rows instead of two, which is a
    different expanded shape to map an update through. The disk bounds
    ``min x1``, so the model stays bounded.
    """
    n = 2
    model = data_model.DataModel()
    # row 0: x1 + x2 >= b0 ; row 1: x1 - x2 <= b1
    model.set_csr_constraint_matrix(
        np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float64),
        np.array([0, 1, 0, 1], dtype=np.int32),
        np.array([0, 2, 4], dtype=np.int32),
    )
    model.set_constraint_bounds(np.asarray(rhs, dtype=np.float64))
    model.set_row_types("GL")
    model.set_objective_coefficients(np.asarray(objective, dtype=np.float64))
    model.set_variable_lower_bounds(np.full(n, -np.inf))
    model.set_variable_upper_bounds(np.full(n, np.inf))
    model.add_quadratic_constraint(
        vals=np.array([1.0, 1.0]),
        rows=np.array([0, 1], dtype=np.int32),
        cols=np.array([0, 1], dtype=np.int32),
        linear_values=np.array([2.0]),
        linear_indices=np.array([0], dtype=np.int32),
        rhs_value=8.0,
        sense="L",
    )
    return model


def _full_cone_solve(build, rhs, objective, **kwargs):
    """Oracle: a fresh cone model with default settings, no cache in play."""
    return solver.Solve(
        build(rhs, objective, **kwargs), solver_settings.SolverSettings()
    )


# Builder, objective, first RHS, then the RHSs to update to. The updates stay
# feasible: on the shifted disk, x1 + x2 tops out just above 2.16.
CONE_CASES = {
    "lorentz": (
        _build_lorentz,
        [1.0, 0.0, 0.0],
        [2.0, 9.0],
        [[3.0, 9.0], [1.5, 9.0]],
    ),
    "general": (
        _build_general_cone,
        [1.0, 0.0],
        [1.0, 5.0],
        [[1.5, 5.0], [0.5, 5.0]],
    ),
}


@pytest.mark.parametrize("case", list(CONE_CASES))
def test_cone_update_rhs_matches_full_solve(case, capfd):
    """An RHS update on a cone model must agree with a fresh full solve.

    The model RHS has two entries while the converted problem has more rows,
    so this fails outright if the update is not mapped into the expanded
    layout, and gives a wrong answer if the appended rows lose their own RHS.
    """
    build, objective, first_rhs, later_rhs = CONE_CASES[case]
    settings = _sequence_settings()
    model = build(first_rhs, objective)

    first, _ = _solve(model, settings, capfd)
    assert first.get_termination_reason() == "Optimal"

    for rhs in later_rhs:
        model.update_rhs(np.asarray(rhs, dtype=np.float64))
        reused, log = _solve(model, settings, capfd)
        assert REUSE_LOG in log, "update_rhs fell back to a full solve"
        _assert_matches_oracle(reused, _full_cone_solve(build, rhs, objective))


@pytest.mark.parametrize("case", list(CONE_CASES))
def test_cone_update_rhs_moves_the_optimum(case):
    """Guard the oracles: the RHS being updated has to matter.

    Without this, a cached solve that ignored the new RHS would still match an
    oracle that ignored it too, and every comparison above would pass.
    """
    build, objective, first_rhs, later_rhs = CONE_CASES[case]
    objectives = {
        _full_cone_solve(build, rhs, objective).get_primal_objective()
        for rhs in [first_rhs] + later_rhs
    }
    assert len(objectives) == 1 + len(later_rhs)


def test_lorentz_optimum_is_the_expected_one():
    """Pin one closed form, so the oracles are not the only reference.

    ``||(x1, x2)|| <= t`` with ``x1 + x2 >= b`` and ``x >= 0`` splits the sum
    evenly, putting the optimum of ``min t`` at ``b / sqrt(2)``.
    """
    for b in (2.0, 3.0):
        solution = _full_cone_solve(_build_lorentz, [b, 9.0], [1.0, 0.0, 0.0])
        assert solution.get_termination_reason() == "Optimal"
        assert solution.get_primal_objective() == pytest.approx(
            b / np.sqrt(2.0), rel=1e-5, abs=1e-5
        )


def test_cone_update_linear_objective_matches_full_solve(capfd):
    """The conversion permutes columns, so the objective needs remapping too."""
    settings = _sequence_settings()
    rhs = [2.0, 9.0]
    model = _build_lorentz(rhs, [1.0, 0.0, 0.0])

    first, _ = _solve(model, settings, capfd)
    assert first.get_termination_reason() == "Optimal"

    new_objective = [1.0, 0.25, 0.0]
    model.update_linear_objective(np.asarray(new_objective, dtype=np.float64))
    reused, log = _solve(model, settings, capfd)
    assert REUSE_LOG in log, (
        "update_linear_objective fell back to a full solve"
    )
    _assert_matches_oracle(
        reused, _full_cone_solve(_build_lorentz, rhs, new_objective)
    )


def test_cone_update_rhs_rejects_lost_cone_head_bound(capfd):
    """A cone head proved nonnegative by a row cannot lose that proof.

    The head here is free, so the conversion only accepts the model because
    row 1 forces t >= 0. An RHS that relaxes the row to t >= -1 makes this a
    model a full solve refuses, and reuse has to refuse it the same way rather
    than solving a stale cone formulation.
    """
    settings = _sequence_settings()
    objective = [1.0, 0.0, 0.0]
    model = _build_lorentz([2.0, 0.0], objective, cone_head_free=True)

    first, _ = _solve(model, settings, capfd)
    assert first.get_termination_reason() == "Optimal"

    with pytest.raises(Exception, match="nonnegative"):
        model.update_rhs(np.array([2.0, -1.0]))

    # A full solve of the same model does not return an optimum either,
    # whether it reports the rejection as an exception or a status.
    try:
        oracle = _full_cone_solve(
            _build_lorentz, [2.0, -1.0], objective, cone_head_free=True
        )
    except Exception:
        pass
    else:
        assert oracle.get_termination_reason() != "Optimal"


def test_cone_update_rhs_rejects_wrong_length(capfd):
    """Length is validated against the model rows, not the converted rows."""
    settings = _sequence_settings()
    model = _build_lorentz([2.0, 9.0], [1.0, 0.0, 0.0])
    first, _ = _solve(model, settings, capfd)
    assert first.get_termination_reason() == "Optimal"

    # Two model rows. Passing the converted row count must not be accepted.
    with pytest.raises(Exception, match="match the cached model row count"):
        model.update_rhs(np.array([2.0, 9.0, 0.0, 0.0]))


def test_update_rhs_rejects_wrong_length():
    """Length is validated against the cached user row count."""
    settings = _sequence_settings()
    model = _build(**dict(MIXED_SENSES, rhs=[5.0, 8.0, 3.0]))
    assert solver.Solve(model, settings).get_termination_reason() == "Optimal"

    with pytest.raises(Exception, match="match the cached model row count"):
        model.update_rhs(np.array([1.0, 2.0]))
