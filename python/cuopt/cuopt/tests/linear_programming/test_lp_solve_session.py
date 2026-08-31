# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Solver-session cache tests: ADAT symbolic reuse and sparsity-hash mismatch paths.

Requires a CUDA GPU and ``CUOPT_CACHE_PROFILE=1`` for C07 timing assertions
(log-based checks are used when reuse lines are captured).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

from cuopt.linear_programming.problem import LinearExpression, MINIMIZE

_helpers_path = Path(__file__).resolve().with_name("session_cache_helpers.py")
_spec = importlib.util.spec_from_file_location("session_cache_helpers", _helpers_path)
assert _spec and _spec.loader
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)

assert_full_symbolic_reanalyze = _helpers.assert_full_symbolic_reanalyze
assert_optimal = _helpers.assert_optimal
assert_warm_symbolic_reuse = _helpers.assert_warm_symbolic_reuse
build_augmented_qp = _helpers.build_augmented_qp
build_sparse_lp = _helpers.build_sparse_lp
count_log_matches = _helpers.count_log_matches
perturb_lp_values = _helpers.perturb_lp_values
perturb_qp_values = _helpers.perturb_qp_values
rewire_lp_row_sparsity = _helpers.rewire_lp_row_sparsity
session_barrier_settings = _helpers.session_barrier_settings
solve_with_log = _helpers.solve_with_log
stored_sparsity_hashes = _helpers.stored_sparsity_hashes
_CLEAR_CACHE_LINE = _helpers._CLEAR_CACHE_LINE
_REUSE_SYMBOLIC_LINE = _helpers._REUSE_SYMBOLIC_LINE

os.environ.setdefault("CUOPT_CACHE_PROFILE", "1")


def _resolve_after_cache_clear(prob, settings, session):
    """If adopt hits a stale hash and clears cache, one retry must succeed."""
    sol, log_text, profile = solve_with_log(prob, settings, session=session)
    if count_log_matches(log_text, _CLEAR_CACHE_LINE) >= 1:
        sol, log_text, profile = solve_with_log(prob, settings, session=session)
    return sol, log_text, profile


def test_adat_session_warm_reuse():
    """Sparse LP on ADAT path: store on cold, reuse on value-only warm solve."""
    prob, xs, c = build_sparse_lp(seed=7)
    settings = session_barrier_settings(session_enabled=True, augmented=0)

    sol_cold, cold_log, cold_profile = solve_with_log(prob, settings)
    assert_optimal(sol_cold)
    session = sol_cold.lp_solve_session
    assert session is not None

    perturb_lp_values(prob, xs, c, seed=101)
    sol_warm, warm_log, warm_profile = solve_with_log(prob, settings, session=session)
    assert_optimal(sol_warm)

    assert_warm_symbolic_reuse(
        cold_log, warm_log, cold_profile, warm_profile, expect_adat=True
    )


def test_augmented_session_warm_reuse():
    """QP with off-diagonal Q (augmented KKT): store on cold, reuse on warm solve."""
    prob, xs, c = build_augmented_qp(seed=11)
    settings = session_barrier_settings(session_enabled=True, augmented=-1)

    sol_cold, cold_log, cold_profile = solve_with_log(prob, settings)
    assert_optimal(sol_cold)
    session = sol_cold.lp_solve_session

    perturb_qp_values(prob, xs, c, seed=202)
    sol_warm, warm_log, warm_profile = solve_with_log(prob, settings, session=session)
    assert_optimal(sol_warm)

    assert_warm_symbolic_reuse(
        cold_log, warm_log, cold_profile, warm_profile, expect_augmented=True
    )


def test_sparsity_hash_mismatch_add_constraint():
    """Adding a constraint changes sparsity; solver must not reuse symbolic cache."""
    prob, xs, c = build_sparse_lp(seed=13)
    settings = session_barrier_settings(session_enabled=True, augmented=0)

    sol_cold, cold_log, cold_profile = solve_with_log(prob, settings)
    assert_optimal(sol_cold)
    session = sol_cold.lp_solve_session
    cold_c07 = cold_profile.get("C07", 0.0)
    cold_hashes = stored_sparsity_hashes(cold_log)

    perturb_lp_values(prob, xs, c, seed=303)
    prob.addConstraint(
        LinearExpression([xs[0], xs[1]], [1.0, 1.0], 0.0) <= 5.0, name="extra_cap"
    )
    sol_warm, warm_log, warm_profile = _resolve_after_cache_clear(
        prob, settings, session
    )
    assert_optimal(sol_warm)
    assert count_log_matches(warm_log, _REUSE_SYMBOLIC_LINE) == 0
    warm_hash = stored_sparsity_hashes(warm_log)
    if warm_hash and cold_hashes:
        assert warm_hash[-1] != cold_hashes[-1]
    assert_full_symbolic_reanalyze(warm_log, warm_profile, cold_c07=cold_c07)


def test_sparsity_hash_mismatch_rewire_row_pattern():
    """Same m/n but a different A row pattern must not reuse symbolic cache."""
    prob, xs, c = build_sparse_lp(seed=37)
    settings = session_barrier_settings(session_enabled=True, augmented=0)

    sol_cold, cold_log, cold_profile = solve_with_log(prob, settings)
    assert_optimal(sol_cold)
    n_before = len(prob.vars)
    m_before = len(prob.constrs)
    session = sol_cold.lp_solve_session
    cold_c07 = cold_profile.get("C07", 0.0)
    cold_hashes = stored_sparsity_hashes(cold_log)

    rewire_lp_row_sparsity(prob, xs, row_idx=0, seed=707)
    assert len(prob.vars) == n_before
    assert len(prob.constrs) == m_before

    sol_warm, warm_log, warm_profile = _resolve_after_cache_clear(
        prob, settings, session
    )
    assert_optimal(sol_warm)
    assert count_log_matches(warm_log, _REUSE_SYMBOLIC_LINE) == 0
    warm_hashes = stored_sparsity_hashes(warm_log)
    if cold_hashes and warm_hashes:
        assert warm_hashes[-1] != cold_hashes[-1]
    assert_full_symbolic_reanalyze(warm_log, warm_profile, cold_c07=cold_c07)


def test_sparsity_hash_mismatch_add_variable():
    """Adding a variable and constraint changes the KKT pattern."""
    prob, xs, c = build_sparse_lp(seed=17)
    settings = session_barrier_settings(session_enabled=True, augmented=0)

    sol_cold, cold_log, cold_profile = solve_with_log(prob, settings)
    assert_optimal(sol_cold)
    session = sol_cold.lp_solve_session
    cold_c07 = cold_profile.get("C07", 0.0)
    cold_hashes = stored_sparsity_hashes(cold_log)

    z = prob.addVariable(lb=0.0, name="z_extra")
    prob.addConstraint(LinearExpression([xs[0], z], [1.0, 1.0], 0.0) <= 10.0)
    c = np.append(c, 3.0)
    prob.setObjective(
        LinearExpression(xs + [z], c.tolist(), 0.0), sense=MINIMIZE
    )

    sol_warm, warm_log, warm_profile = _resolve_after_cache_clear(
        prob, settings, session
    )
    assert_optimal(sol_warm)
    assert count_log_matches(warm_log, _REUSE_SYMBOLIC_LINE) == 0
    warm_hashes = stored_sparsity_hashes(warm_log)
    if cold_hashes and warm_hashes:
        assert warm_hashes[-1] != cold_hashes[-1]
    assert_full_symbolic_reanalyze(warm_log, warm_profile, cold_c07=cold_c07)


def test_cross_system_mismatch_augmented_then_adat():
    """Augmented cache in session must not be reused for a different ADAT problem."""
    prob_qp, _, _ = build_augmented_qp(seed=19)
    settings_aug = session_barrier_settings(session_enabled=True, augmented=-1)
    sol_qp, _, _ = solve_with_log(prob_qp, settings_aug)
    assert_optimal(sol_qp)
    session = sol_qp.lp_solve_session

    prob_lp, xs, c = build_sparse_lp(seed=23)
    settings_adat = session_barrier_settings(session_enabled=True, augmented=0)
    sol_lp, lp_log, lp_profile = _resolve_after_cache_clear(
        prob_lp, settings_adat, session
    )
    assert_optimal(sol_lp)
    assert count_log_matches(lp_log, _REUSE_SYMBOLIC_LINE) == 0
    assert_full_symbolic_reanalyze(lp_log, lp_profile)

    perturb_lp_values(prob_lp, xs, c, seed=404)
    _, warm_log, warm_profile = solve_with_log(prob_lp, settings_adat, session=session)
    assert_warm_symbolic_reuse(
        lp_log, warm_log, lp_profile, warm_profile, expect_adat=True
    )


def test_different_lp_structures_same_session():
    """Two LP instances with different sparsity patterns must not share symbolic cache."""
    prob_a, xs_a, c_a = build_sparse_lp(seed=29, n=400, m=200, nnz_per_row=5)
    prob_b, xs_b, c_b = build_sparse_lp(seed=31, n=900, m=450, nnz_per_row=9)
    settings = session_barrier_settings(session_enabled=True, augmented=0)

    sol_a, cold_log, profile_a = solve_with_log(prob_a, settings)
    assert_optimal(sol_a)
    session = sol_a.lp_solve_session
    cold_c07 = profile_a.get("C07", 0.0)
    cold_hashes = stored_sparsity_hashes(cold_log)

    sol_b, log_b, profile_b = _resolve_after_cache_clear(prob_b, settings, session)
    assert_optimal(sol_b)
    assert count_log_matches(log_b, _REUSE_SYMBOLIC_LINE) == 0
    warm_hashes = stored_sparsity_hashes(log_b)
    if cold_hashes and warm_hashes:
        assert warm_hashes[-1] != cold_hashes[-1]
    assert_full_symbolic_reanalyze(log_b, profile_b, cold_c07=cold_c07)

    perturb_lp_values(prob_b, xs_b, c_b, seed=505)
    _, warm_log, warm_profile = solve_with_log(prob_b, settings, session=session)
    assert_warm_symbolic_reuse(
        log_b, warm_log, profile_b, warm_profile, expect_adat=True
    )
