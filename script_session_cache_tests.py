#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Runnable session-cache checks: ADAT warm reuse and sparsity-hash mismatch.

  CUOPT_CACHE_PROFILE=1 python script_session_cache_tests.py
  CUOPT_SESSION_TEST_SMALL=1 python script_session_cache_tests.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "python",
        "cuopt",
        "cuopt",
        "tests",
        "linear_programming",
    ),
)

from cuopt.linear_programming.problem import LinearExpression
from session_cache_helpers import (  # noqa: E402
    _CLEAR_CACHE_LINE,
    _REUSE_SYMBOLIC_LINE,
    _STORE_ADAT_LINE,
    _STORE_AUGMENTED_LINE,
    assert_full_symbolic_reanalyze,
    assert_optimal,
    assert_warm_symbolic_reuse,
    build_augmented_qp,
    build_sparse_lp,
    count_log_matches,
    perturb_lp_values,
    perturb_qp_values,
    session_barrier_settings,
    solve_with_log,
)

os.environ.setdefault("CUOPT_CACHE_PROFILE", "1")


def _resolve_after_cache_clear(prob, settings, session):
    sol, log_text, profile = solve_with_log(prob, settings, session=session)
    if count_log_matches(log_text, _CLEAR_CACHE_LINE) >= 1:
        sol, log_text, profile = solve_with_log(prob, settings, session=session)
    return sol, log_text, profile


def _run_case(name: str, fn) -> None:
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    t0 = time.perf_counter()
    fn()
    print(f"PASS ({(time.perf_counter() - t0):.1f} s)")


def _case_adat_warm_reuse() -> None:
    prob, xs, c = build_sparse_lp(seed=7)
    settings = session_barrier_settings(session_enabled=True, augmented=0)
    sol_c, cold_log, cold_prof = solve_with_log(prob, settings)
    assert_optimal(sol_c)
    session = sol_c.lp_solve_session
    perturb_lp_values(prob, xs, c, seed=101)
    sol_w, warm_log, warm_prof = solve_with_log(prob, settings, session=session)
    assert_optimal(sol_w)
    assert_warm_symbolic_reuse(cold_log, warm_log, cold_prof, warm_prof, expect_adat=True)
    print(
        f"  cold C07={cold_prof.get('C07', 0):.2f} ms  "
        f"warm C07={warm_prof.get('C07', 0):.2f} ms  "
        f"reuse_logs={count_log_matches(warm_log, _REUSE_SYMBOLIC_LINE)}  "
        f"ADAT_store={count_log_matches(cold_log, _STORE_ADAT_LINE)}"
    )


def _case_augmented_warm_reuse() -> None:
    prob, xs, c = build_augmented_qp(seed=11)
    settings = session_barrier_settings(session_enabled=True, augmented=-1)
    sol_c, cold_log, cold_prof = solve_with_log(prob, settings)
    assert_optimal(sol_c)
    session = sol_c.lp_solve_session
    perturb_qp_values(prob, xs, c, seed=202)
    sol_w, warm_log, warm_prof = solve_with_log(prob, settings, session=session)
    assert_optimal(sol_w)
    assert_warm_symbolic_reuse(
        cold_log, warm_log, cold_prof, warm_prof, expect_augmented=True
    )
    print(
        f"  cold C07={cold_prof.get('C07', 0):.2f} ms  "
        f"warm C07={warm_prof.get('C07', 0):.2f} ms  "
        f"aug_store={count_log_matches(cold_log, _STORE_AUGMENTED_LINE)}"
    )


def _case_hash_mismatch_add_constraint() -> None:
    prob, xs, c = build_sparse_lp(seed=13)
    settings = session_barrier_settings(session_enabled=True, augmented=0)
    sol_c, _, cold_prof = solve_with_log(prob, settings)
    assert_optimal(sol_c)
    session = sol_c.lp_solve_session
    cold_c07 = cold_prof.get("C07", 0.0)
    perturb_lp_values(prob, xs, c, seed=303)
    prob.addConstraint(LinearExpression([xs[0], xs[1]], [1.0, 1.0], 0.0) <= 5.0)
    _, warm_log, warm_prof = _resolve_after_cache_clear(prob, settings, session)
    assert_full_symbolic_reanalyze(warm_log, warm_prof, cold_c07=cold_c07)
    print(
        f"  warm C07={warm_prof.get('C07', 0):.2f} ms (cold ref={cold_c07:.2f})  "
        f"reuse_logs={count_log_matches(warm_log, _REUSE_SYMBOLIC_LINE)}"
    )


def _case_cross_augmented_to_adat() -> None:
    prob_qp, _, _ = build_augmented_qp(seed=19)
    settings_aug = session_barrier_settings(session_enabled=True, augmented=-1)
    sol_qp, _, _ = solve_with_log(prob_qp, settings_aug)
    assert_optimal(sol_qp)
    session = sol_qp.lp_solve_session
    prob_lp, xs, c = build_sparse_lp(seed=23)
    settings_adat = session_barrier_settings(session_enabled=True, augmented=0)
    _, lp_log, lp_prof = _resolve_after_cache_clear(prob_lp, settings_adat, session)
    assert_full_symbolic_reanalyze(lp_log, lp_prof)
    perturb_lp_values(prob_lp, xs, c, seed=404)
    _, warm_log, warm_prof = solve_with_log(prob_lp, settings_adat, session=session)
    assert_warm_symbolic_reuse(lp_log, warm_log, lp_prof, warm_prof, expect_adat=True)
    print("  augmented session -> ADAT LP re-analyzed, then ADAT warm reuse OK")


def main() -> None:
    cases = [
        ("ADAT session warm reuse", _case_adat_warm_reuse),
        ("Augmented QP warm reuse", _case_augmented_warm_reuse),
        ("Sparsity mismatch (add constraint)", _case_hash_mismatch_add_constraint),
        ("Cross-system mismatch (augmented -> ADAT)", _case_cross_augmented_to_adat),
    ]
    print("Session cache verification (GPU required)")
    for name, fn in cases:
        _run_case(name, fn)
    print("\nAll session cache checks passed.")


if __name__ == "__main__":
    main()
