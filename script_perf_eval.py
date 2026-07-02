#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Portfolio QP benchmark for solver-session cache evaluation.

Same problem structure as ``benchmark_cvxpylayers`` (portfolio QP with
``gamma`` quadratic in ``x`` and ``y``, transaction-cost ``z`` linearization).

Measures cold (first) vs warm (value-only) ``prob.solve()`` times on a fixed structure.
Parameter updates are applied outside the timer.
Per-cache breakdown (C01–C09) is measured in cuOpt when ``CUOPT_CACHE_PROFILE=1``
(see ``docs/solver_session_cache.md``).

Environment:
  CUOPT_CACHE_PROFILE=1             -> emit per-cache timings from cuOpt (default on)
  CUOPT_PORTFOLIO_BENCHMARK_SMALL=1  -> k=5, scale=2 (smoke test)
  CUOPT_CACHE_BENCHMARK_WARM=5       -> number of warm iterations (default 5)
  CUOPT_BENCHMARK_MODE=baseline|session|all  -> single mode or two-process run (default all)

Run with conda env ``py313`` activated.

Examples:
  python script_perf_eval.py --mode baseline   # session disabled, one process
  python script_perf_eval.py --mode session    # session enabled, one process
  python script_perf_eval.py --mode all        # baseline then session (new process each)
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from cuopt.linear_programming import solver_settings
from cuopt.linear_programming.problem import (
    LinearExpression,
    MINIMIZE,
    Problem,
    QuadraticExpression,
)
from cuopt.linear_programming.solver.solver_parameters import CUOPT_METHOD
from cuopt.linear_programming.solver_settings import SolverMethod


CACHE_ITEM_NAMES = {
    "C01": "raft::handle_t + stream",
    "C02": "cuBLAS / cuSparse warmup",
    "C03": "Problem fingerprint",
    "C04": "Augmented vs ADAT choice",
    "C05": "ADAT / augmented sparsity pattern",
    "C06": "cuDSS handle + config",
    "C07": "cuDSS symbolic factorization",
    "C08": "Dense-column / SOC layout metadata",
    "C09": "Device buffer allocation sizes",
}

_CACHE_PROFILE_LINE = re.compile(
    r"^Cache profile: (C\d+) .+? ([0-9]+(?:\.[0-9]+)?)\s*$"
)
_REUSE_SYMBOLIC_LINE = re.compile(
    r"Barrier: reusing cuDSS symbolic analysis \(sparsity hash match\)"
)
_REBUILT_SYMBOLIC_LINE = re.compile(
    r"Barrier: rebuilt cuDSS symbolic analysis"
)
_STORE_SYMBOLIC_HASH_LINE = re.compile(
    r"Barrier: stored augmented symbolic cache hash=0x[0-9a-f]+"
)


def _parse_cache_profile(text: str) -> dict[str, float]:
    """Parse the last ``=== Solver cache profile ===`` block from cuOpt logs."""
    profiles: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for line in text.splitlines():
        if "=== Solver cache profile" in line:
            current = {}
            continue
        if line.strip() == "=== End solver cache profile ===":
            if current is not None:
                profiles.append(current)
            current = None
            continue
        if current is None:
            continue
        m = _CACHE_PROFILE_LINE.match(line.strip())
        if m:
            current[m.group(1)] = float(m.group(2))
    return profiles[-1] if profiles else {}


def _avg_cache_profiles(profiles: list[dict[str, float]]) -> dict[str, float]:
    if not profiles:
        return {}
    keys = sorted({k for p in profiles for k in p})
    return {k: float(np.mean([p.get(k, 0.0) for p in profiles])) for k in keys}


@contextmanager
def _capture_solver_output():
    """Capture C++ solver logs written directly to stderr (fd 2)."""
    read_fd, write_fd = os.pipe()
    saved_stderr = os.dup(2)
    capture = io.StringIO()
    try:
        os.dup2(write_fd, 2)
        os.close(write_fd)
        yield capture
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        with os.fdopen(read_fd, "r", encoding="utf-8", errors="replace") as reader:
            text = reader.read()
        capture.write(text)
        capture.seek(0)
        if text:
            sys.stderr.write(text)
            sys.stderr.flush()


def _objective_snapshot(
    prob: Problem,
    info: dict,
    solution,
    mu: np.ndarray,
    d: float,
    x0: np.ndarray,
    tc: float,
    label: str,
) -> dict:
    x_np = portfolio_x(prob)
    eval_info = {**info, "mu": mu, "x0": x0, "d": d, "tc_rate": tc}
    formula_obj = float(portfolio_objective(eval_info, x_np))
    cuopt_obj = float(solution.get_primal_objective())
    return {
        "label": label,
        "cuopt_primal_objective": cuopt_obj,
        "formula_objective": formula_obj,
        "cuopt_vs_formula_delta": abs(cuopt_obj - formula_obj),
        "termination": str(solution.termination_status),
        "x_norm": float(np.linalg.norm(x_np)),
    }


def timed_solve(
    prob: Problem,
    settings,
    info: dict,
    mu: np.ndarray,
    d: float,
    x0: np.ndarray,
    tc: float,
    label: str,
) -> tuple[float, dict[str, float], str, dict]:
    """Run ``prob.solve``; return elapsed ms, cache profile, log text, objective snapshot."""
    t0 = time.perf_counter()
    with _capture_solver_output() as capture:
        solution = prob.solve(settings)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log_text = capture.getvalue()
    profile = _parse_cache_profile(log_text)
    snapshot = _objective_snapshot(prob, info, solution, mu, d, x0, tc, label)
    return elapsed_ms, profile, log_text, snapshot


def _count_log_matches(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def generate_portfolio_data(k=50, scale=100, seed=42):
    """Identical to ``benchmark_cvxpylayers.generate_portfolio_data``."""
    np.random.seed(seed)
    n = k * scale
    d_diag = np.random.rand(n) * np.sqrt(k)
    f_mat = np.random.randn(n, k) * (np.random.rand(n, k) < 0.5)
    omega_temp = np.random.randn(k, k)
    omega = (omega_temp @ omega_temp.T) / k
    mu = (3.0 + 9.0 * np.random.rand(n)) / 100.0
    x0 = np.zeros(n)
    gamma = 1.0
    d = 1.0
    tc_rate = 0.002
    return {
        "n": n,
        "k": k,
        "D_diag": d_diag,
        "F": f_mat,
        "Omega": omega,
        "mu": mu,
        "x0": x0,
        "gamma": gamma,
        "d": d,
        "tc_rate": tc_rate,
    }


def portfolio_objective(info, x_np):
    """Same evaluation as ``benchmark_cvxpylayers.portfolio_objective``."""
    y = info["F"].T @ x_np
    z = np.abs(x_np - info["x0"])
    d_matrix = np.diag(info["D_diag"])
    return (
        -info["mu"] @ x_np
        + info["gamma"]
        * (x_np @ d_matrix @ x_np + y @ info["Omega"] @ y)
        + info["tc_rate"] * np.sum(z)
    )


def build_cuopt_portfolio_problem(info: dict) -> Problem:
    """Build portfolio QP and stash handles for value-only updates."""
    n = info["n"]
    k = info["k"]
    gamma = info["gamma"]
    d_diag = info["D_diag"]
    F = info["F"]
    omega = info["Omega"]
    mu = info["mu"]
    x0 = info["x0"]
    d_rhs = float(info["d"])
    tc_rate = float(info["tc_rate"])

    prob = Problem("portfolio_qp_cuopt")
    xs = [prob.addVariable(lb=0.0, ub=0.1, name=f"x{i}") for i in range(n)]
    ys = [prob.addVariable(lb=0.0, ub=0.1, name=f"y{j}") for j in range(k)]
    zs = [prob.addVariable(lb=0.0, ub=1.0e6, name=f"z{i}") for i in range(n)]

    budget_constr = prob.addConstraint(LinearExpression(xs, [1.0] * n, 0.0) == d_rhs)

    for j in range(k):
        c = [-float(F[i, j]) for i in range(n)] + [1.0]
        v = xs + [ys[j]]
        prob.addConstraint(LinearExpression(v, c, 0.0) == 0.0)

    x0_le_constrs = []
    x0_ge_constrs = []
    for i in range(n):
        x0i = float(x0[i])
        x0_le_constrs.append(
            prob.addConstraint(LinearExpression([xs[i], zs[i]], [1.0, -1.0], 0.0) <= x0i)
        )
        x0_ge_constrs.append(
            prob.addConstraint(LinearExpression([xs[i], zs[i]], [1.0, 1.0], 0.0) >= x0i)
        )

    qv1, qv2, qc = [], [], []
    for i in range(n):
        qv1.append(xs[i])
        qv2.append(xs[i])
        qc.append(gamma * float(d_diag[i]))
    for p in range(k):
        for q in range(k):
            o = float(omega[p, q])
            if o != 0.0:
                qv1.append(ys[p])
                qv2.append(ys[q])
                qc.append(gamma * o)

    quad = QuadraticExpression(
        qvars1=qv1,
        qvars2=qv2,
        qcoefficients=qc,
        vars=[],
        coefficients=[],
        constant=0.0,
    )
    lin_vars = xs + ys + zs
    lin_c = np.concatenate([-mu, np.zeros(k, dtype=np.float64), np.full(n, tc_rate)])
    linear = LinearExpression(lin_vars, lin_c.tolist(), 0.0)
    prob.setObjective(quad + linear, sense=MINIMIZE)

    prob._portfolio_n = n
    prob._portfolio_k = k
    prob._portfolio_xs = xs
    prob._portfolio_zs = zs
    prob._portfolio_budget_constr = budget_constr
    prob._portfolio_x0_le = x0_le_constrs
    prob._portfolio_x0_ge = x0_ge_constrs
    # Keep Q matrix: Problem.reset_solved_values() clears objective_qmatrix.
    prob._portfolio_qmatrix = prob.objective_qmatrix
    return prob


def _invalidate_for_resolve(prob: Problem) -> None:
    """Mark problem dirty for re-solve without dropping structure or DataModel."""
    prob.solved = False
    prob.warmstart_data = None
    prob.objective_qmatrix = prob._portfolio_qmatrix


def update_portfolio_values(
    prob: Problem,
    mu: np.ndarray,
    d: float,
    x0: np.ndarray,
    tc: float,
) -> None:
    """Value-only update: linear objective coeffs and selected constraint RHS."""
    n = prob._portfolio_n
    for i in range(n):
        prob._portfolio_xs[i].setObjectiveCoefficient(-float(mu[i]))
        prob._portfolio_zs[i].setObjectiveCoefficient(float(tc))
        prob._portfolio_x0_le[i].RHS = float(x0[i])
        prob._portfolio_x0_ge[i].RHS = float(x0[i])
    prob._portfolio_budget_constr.RHS = float(d)
    _invalidate_for_resolve(prob)


def portfolio_x(prob: Problem) -> np.ndarray:
    """First ``n`` primal values (portfolio weights ``x``)."""
    n = prob._portfolio_n
    return np.array([float(prob.vars[i].Value) for i in range(n)])


def solve_portfolio(
    prob: Problem,
    mu: np.ndarray,
    d: float,
    x0: np.ndarray,
    tc: float,
    settings,
) -> np.ndarray:
    """Update values and solve; returns primal ``x`` (first n variables)."""
    update_portfolio_values(prob, mu, d, x0, tc)
    prob.solve(settings)
    return portfolio_x(prob)


def _perturb_portfolio_params(info: dict, seed: int) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Small value-only perturbation; structure of the Problem is unchanged."""
    rng = np.random.default_rng(seed)
    mu = info["mu"] * (1.0 + 0.001 * rng.standard_normal(info["n"]))
    d = float(info["d"]) * (1.0 + 0.001 * rng.standard_normal())
    x0 = info["x0"] + 1e-4 * rng.standard_normal(info["n"])
    tc = float(info["tc_rate"]) * (1.0 + 0.001 * rng.standard_normal())
    return mu, d, x0, tc


def _base_portfolio_params(info: dict) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Unperturbed parameters from ``info``."""
    return info["mu"], float(info["d"]), info["x0"], float(info["tc_rate"])


def _fresh_settings(*, use_session: bool) -> solver_settings.SolverSettings:
    ss = solver_settings.SolverSettings()
    ss.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)
    if use_session:
        ss.set_session_enabled(True)
    return ss


def run_cache_benchmark(
    prob: Problem,
    info: dict,
    settings,
    n_warm: int = 5,
    *,
    same_values: bool = False,
    use_session: bool = False,
):
    """
    Cold (first) vs warm (subsequent) solve times.

    If ``same_values`` is True, every solve uses identical ``mu, d, x0, tc``
    (no parameter perturbation). Otherwise warm solves use perturbed values.

    Timings cover ``prob.solve()`` only; parameter updates run outside the timer.
    """
    if same_values:
        mu, d, x0, tc = _base_portfolio_params(info)
    else:
        mu, d, x0, tc = _perturb_portfolio_params(info, seed=0)

    if use_session:
        prob._session = None

    update_portfolio_values(prob, mu, d, x0, tc)
    cold_ms, cold_profile, cold_log, cold_obj = timed_solve(
        prob, settings, info, mu, d, x0, tc, "cold"
    )
    cold_reuse = _count_log_matches(cold_log, _REUSE_SYMBOLIC_LINE)
    cold_rebuild = _count_log_matches(cold_log, _REBUILT_SYMBOLIC_LINE)
    cold_store_hash = _count_log_matches(cold_log, _STORE_SYMBOLIC_HASH_LINE)
    objective_records = [cold_obj]

    warm_ms = []
    warm_profiles = []
    warm_reuse_hits = []
    warm_rebuild_hits = []
    warm_store_hash_hits = []
    for i in range(n_warm):
        if not same_values:
            mu, d, x0, tc = _perturb_portfolio_params(info, seed=100 + i)
        update_portfolio_values(prob, mu, d, x0, tc)
        ms, profile, log_text, obj_snap = timed_solve(
            prob, settings, info, mu, d, x0, tc, f"warm_{i}"
        )
        warm_ms.append(ms)
        objective_records.append(obj_snap)
        warm_reuse_hits.append(_count_log_matches(log_text, _REUSE_SYMBOLIC_LINE))
        warm_rebuild_hits.append(_count_log_matches(log_text, _REBUILT_SYMBOLIC_LINE))
        warm_store_hash_hits.append(_count_log_matches(log_text, _STORE_SYMBOLIC_HASH_LINE))
        if profile:
            warm_profiles.append(profile)

    warm_avg = float(np.mean(warm_ms))
    warm_min = float(np.min(warm_ms))
    saved_ms = cold_ms - warm_avg
    saved_pct = (saved_ms / cold_ms * 100.0) if cold_ms > 0 else 0.0
    warm_profile_avg = _avg_cache_profiles(warm_profiles)
    cache_save_ms = {}
    for cid in CACHE_ITEM_NAMES:
        cold_val = cold_profile.get(cid, 0.0)
        warm_val = warm_profile_avg.get(cid, 0.0)
        cache_save_ms[cid] = max(cold_val - warm_val, 0.0)
    return {
        "cold_ms": cold_ms,
        "warm_ms": warm_ms,
        "warm_avg_ms": warm_avg,
        "warm_min_ms": warm_min,
        "saved_ms": saved_ms,
        "saved_pct": saved_pct,
        "same_values": same_values,
        "use_session": use_session,
        "cold_cache_ms": cold_profile,
        "warm_cache_ms_avg": warm_profile_avg,
        "cache_save_ms": cache_save_ms,
        "cold_reuse_log_count": cold_reuse,
        "cold_rebuild_log_count": cold_rebuild,
        "cold_store_hash_count": cold_store_hash,
        "warm_reuse_log_counts": warm_reuse_hits,
        "warm_rebuild_log_counts": warm_rebuild_hits,
        "warm_store_hash_counts": warm_store_hash_hits,
        "session_after_cold": prob._session is not None,
        "objective_records": objective_records,
    }


def _param_schedule(info: dict, n_warm: int, *, same_values: bool) -> list[tuple[str, np.ndarray, float, np.ndarray, float]]:
    schedule: list[tuple[str, np.ndarray, float, np.ndarray, float]] = []
    if same_values:
        mu, d, x0, tc = _base_portfolio_params(info)
        for label in ["cold"] + [f"warm_{i}" for i in range(n_warm)]:
            schedule.append((label, mu, d, x0, tc))
    else:
        mu, d, x0, tc = _perturb_portfolio_params(info, seed=0)
        schedule.append(("cold", mu, d, x0, tc))
        for i in range(n_warm):
            mu, d, x0, tc = _perturb_portfolio_params(info, seed=100 + i)
            schedule.append((f"warm_{i}", mu, d, x0, tc))
    return schedule


def _solve_one_objective(
    info: dict,
    settings,
    label: str,
    mu: np.ndarray,
    d: float,
    x0: np.ndarray,
    tc: float,
    *,
    use_session: bool,
    session=None,
) -> dict:
    prob = build_cuopt_portfolio_problem(info)
    if use_session and session is not None:
        prob._session = session
    update_portfolio_values(prob, mu, d, x0, tc)
    with _capture_solver_output():
        solution = prob.solve(settings, session=session if use_session else None)
    snap = _objective_snapshot(prob, info, solution, mu, d, x0, tc, label)
    snap["use_session"] = use_session
    snap["session_capsule"] = solution.lp_solve_session if use_session else None
    return snap


def verify_objectives_across_modes(
    info: dict,
    n_warm: int,
    *,
    same_values: bool,
    obj_atol: float = 1e-6,
) -> dict:
    """
    For each parameter setting in the benchmark schedule, solve once without
    session and once with session (cold then warm reuse) and compare objectives.
    """
    schedule = _param_schedule(info, n_warm, same_values=same_values)
    ss_base = _fresh_settings(use_session=False)
    ss_sess = _fresh_settings(use_session=True)

    paired = []
    session = None
    for label, mu, d, x0, tc in schedule:
        base = _solve_one_objective(
            info, ss_base, label, mu, d, x0, tc, use_session=False
        )
        if label == "cold":
            sess = _solve_one_objective(
                info, ss_sess, label, mu, d, x0, tc, use_session=True
            )
            session = sess.get("session_capsule")
        else:
            sess = _solve_one_objective(
                info,
                ss_sess,
                label,
                mu,
                d,
                x0,
                tc,
                use_session=True,
                session=session,
            )
        delta = abs(base["cuopt_primal_objective"] - sess["cuopt_primal_objective"])
        paired.append(
            {
                "label": label,
                "baseline_cuopt": base["cuopt_primal_objective"],
                "session_cuopt": sess["cuopt_primal_objective"],
                "baseline_formula": base["formula_objective"],
                "session_formula": sess["formula_objective"],
                "cuopt_delta": delta,
                "cuopt_match": delta <= obj_atol,
                "baseline_termination": base["termination"],
                "session_termination": sess["termination"],
            }
        )

    all_cuopt_match = all(r["cuopt_match"] for r in paired)
    return {
        "paired": paired,
        "all_cuopt_match": all_cuopt_match,
        "obj_atol": obj_atol,
    }


def _print_objective_records(mode: str, records: list[dict]) -> None:
    print(f"\n  Objectives ({mode}):")
    print(
        "    {:>8}  {:>18}  {:>18}  {:>12}  {}".format(
            "run", "cuOpt primal", "formula", "cuOpt-form", "status"
        )
    )
    for rec in records:
        print(
            "    {:>8}  {:>18.10f}  {:>18.10f}  {:>12.2e}  {}".format(
                rec["label"],
                rec["cuopt_primal_objective"],
                rec["formula_objective"],
                rec["cuopt_vs_formula_delta"],
                rec["termination"],
            )
        )


def _print_objective_cross_check(cross: dict) -> None:
    print("\n" + "=" * 50)
    print("Objective cross-check (baseline vs session, same parameters)")
    print("=" * 50)
    print(
        f"  Tolerance (cuOpt primal): {cross['obj_atol']:.1e}  "
        f"Overall: {'PASS' if cross['all_cuopt_match'] else 'FAIL'}"
    )
    print(
        "    {:>8}  {:>18}  {:>18}  {:>12}  {}".format(
            "run", "baseline cuOpt", "session cuOpt", "delta", "match"
        )
    )
    for row in cross["paired"]:
        mark = "OK" if row["cuopt_match"] else "FAIL"
        print(
            "    {:>8}  {:>18.10f}  {:>18.10f}  {:>12.2e}  {}".format(
                row["label"],
                row["baseline_cuopt"],
                row["session_cuopt"],
                row["cuopt_delta"],
                mark,
            )
        )


def _write_results(
    results_path: str,
    *,
    k: int,
    scale: int,
    n_assets: int,
    n_warm: int,
    build_time_s: float,
    bench: dict,
) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    cold_cache = bench.get("cold_cache_ms") or {}
    warm_cache = bench.get("warm_cache_ms_avg") or {}
    cache_save = bench.get("cache_save_ms") or {}
    has_measured_cache = bool(cold_cache or warm_cache)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("# Cache performance results\n\n")
        f.write("Generated by `script_perf_eval.py` (conda env `py313`).\n\n")
        if has_measured_cache:
            f.write(
                "Per-cache rows are measured by cuOpt (`CUOPT_CACHE_PROFILE=1`) as "
                "cold minus warm average component time.\n\n"
            )
        else:
            f.write(
                "**Note:** Rebuild cuOpt with cache profiling enabled and set "
                "`CUOPT_CACHE_PROFILE=1` for per-cache rows.\n\n"
            )
        f.write(f"- Problem: portfolio QP, k={k}, scale={scale}, n={n_assets}\n")
        f.write(f"- Solver: Barrier (QP)\n")
        f.write(f"- Parameters: {'identical every solve' if bench.get('same_values') else 'perturbed on warm solves'}\n")
        f.write(f"- Warm iterations: {n_warm}\n\n")
        f.write("## Measured (aggregate)\n\n")
        f.write("| Metric | ms |\n|--------|----|\n")
        f.write(f"| Structure build | {build_time_s * 1000:.2f} |\n")
        f.write(f"| Cold solve (1st) | {bench['cold_ms']:.2f} |\n")
        f.write(f"| Warm solve (avg) | {bench['warm_avg_ms']:.2f} |\n")
        f.write(f"| Warm solve (min) | {bench['warm_min_ms']:.2f} |\n")
        f.write(
            f"| **Aggregate save** | **{bench['saved_ms']:.2f}** "
            f"({bench['saved_pct']:.1f}%) |\n\n"
        )
        f.write("## Per-cache component times (ms)\n\n")
        f.write(
            "| ID | Cache item | Cold | Warm (avg) | Est. save (cold−warm) | "
            "% of cold solve |\n"
        )
        f.write("|----|------------|------|------------|----------------------|"
                "-----------------|\n")
        for cid, name in CACHE_ITEM_NAMES.items():
            cold_ms = cold_cache.get(cid, 0.0)
            warm_ms = warm_cache.get(cid, 0.0)
            save_ms = cache_save.get(cid, 0.0)
            pct = (save_ms / bench["cold_ms"] * 100.0) if bench["cold_ms"] > 0 else 0.0
            f.write(
                f"| {cid} | {name} | {cold_ms:.2f} | {warm_ms:.2f} | "
                f"{save_ms:.2f} | {pct:.1f}% |\n"
            )
        f.write("\nSee `docs/solver_session_cache.md` for cache definitions.\n")


def _print_bench_summary(label: str, bench: dict, n_warm: int) -> None:
    print(f"\n{label}")
    print("-" * 50)
    print(f"  session_enabled:    {bench.get('use_session', False)}")
    print(f"  session after cold: {bench.get('session_after_cold', False)}")
    print(f"  Cold solve (1st):     {bench['cold_ms']:.2f} ms")
    print(f"  Warm solve (avg):     {bench['warm_avg_ms']:.2f} ms  (n={n_warm})")
    print(f"  Warm solve (best):    {bench['warm_min_ms']:.2f} ms")
    print(f"  Aggregate save:       {bench['saved_ms']:.2f} ms ({bench['saved_pct']:.1f}%)")
    c07_c = bench.get("cold_cache_ms", {}).get("C07", 0.0)
    c07_w = bench.get("warm_cache_ms_avg", {}).get("C07", 0.0)
    print(f"  C07 symbolic (cold/warm avg): {c07_c:.2f} / {c07_w:.2f} ms")
    print(
        f"  Sparsity hash reuse logs: cold={bench.get('cold_reuse_log_count', 0)}, "
        f"warm={bench.get('warm_reuse_log_counts', [])}"
    )
    print(
        f"  Symbolic rebuild logs:    cold={bench.get('cold_rebuild_log_count', 0)}, "
        f"warm={bench.get('warm_rebuild_log_counts', [])}"
    )
    print(
        f"  Store hash logs:          cold={bench.get('cold_store_hash_count', 0)}, "
        f"warm={bench.get('warm_store_hash_counts', [])}"
    )
    if bench.get("use_session"):
        warm_reuse = bench.get("warm_reuse_log_counts", [])
        if warm_reuse and all(c >= 1 for c in warm_reuse):
            print("  Fingerprint/sparsity gate: PASS (warm runs reused symbolic analysis)")
        elif warm_reuse and any(c >= 1 for c in warm_reuse):
            print("  Fingerprint/sparsity gate: PARTIAL (some warm runs reused symbolic)")
        else:
            print("  Fingerprint/sparsity gate: FAIL (no warm reuse log; check session wiring)")
    if bench.get("objective_records"):
        _print_objective_records(
            "session" if bench.get("use_session") else "baseline",
            bench["objective_records"],
        )


def _assert_objectives(records: list[dict], *, formula_atol: float) -> None:
    for rec in records:
        if "Optimal" not in rec["termination"] and rec["termination"] != "1":
            raise AssertionError(
                f"{rec['label']}: expected Optimal termination, got {rec['termination']}"
            )
        if rec["cuopt_vs_formula_delta"] > formula_atol:
            raise AssertionError(
                f"{rec['label']}: cuOpt vs formula delta {rec['cuopt_vs_formula_delta']:.2e} "
                f"> {formula_atol:.1e}"
            )


def _assert_log_expectations(mode: str, bench: dict) -> None:
    warm_reuse = bench.get("warm_reuse_log_counts", [])
    c07_c = bench.get("cold_cache_ms", {}).get("C07", 0.0)
    c07_w = bench.get("warm_cache_ms_avg", {}).get("C07", 0.0)

    if mode == "baseline":
        if bench.get("cold_reuse_log_count", 0) != 0 or any(c != 0 for c in warm_reuse):
            raise AssertionError(
                f"baseline: unexpected sparsity reuse log "
                f"(cold={bench.get('cold_reuse_log_count')}, warm={warm_reuse})"
            )
        if c07_w < 50.0:
            raise AssertionError(
                f"baseline warm: expected C07 symbolic > 50 ms without session, got {c07_w:.2f}"
            )
        print(
            f"  Log/fingerprint check: PASS (no reuse logs; warm C07={c07_w:.1f} ms each solve)"
        )
    elif mode == "session":
        if bench.get("cold_reuse_log_count", 0) != 0:
            raise AssertionError("session cold: unexpected sparsity reuse log on first solve")
        if c07_c < 50.0:
            raise AssertionError(
                f"session cold: expected C07 symbolic on first solve, got {c07_c:.2f} ms"
            )
        if c07_w > 1.0:
            raise AssertionError(
                f"session warm: expected C07 symbolic ~0 ms with reuse, got {c07_w:.2f} ms"
            )
        if any(c > 0 for c in warm_reuse):
            if not all(c >= 1 for c in warm_reuse):
                raise AssertionError(
                    f"session warm: partial reuse logs in capture {warm_reuse}"
                )
            print(
                f"  Log/fingerprint check: PASS (reuse logs on all warm; "
                f"C07 cold={c07_c:.1f} ms warm avg={c07_w:.2f} ms)"
            )
        else:
            print(
                f"  Log/fingerprint check: PASS via C07 "
                f"(cold={c07_c:.1f} ms, warm avg={c07_w:.2f} ms; "
                "barrier reuse lines may be stderr-buffered when piped)"
            )
    else:
        raise ValueError(f"unknown benchmark mode {mode!r}")


def _write_bench_artifact(path: Path, mode: str, bench: dict) -> None:
    payload = {
        "mode": mode,
        "objective_records": bench["objective_records"],
        "cold_ms": bench["cold_ms"],
        "warm_avg_ms": bench["warm_avg_ms"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compare_mode_objectives(
    baseline_path: Path,
    session_path: Path,
    *,
    obj_atol: float,
) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    session = json.loads(session_path.read_text(encoding="utf-8"))
    print("\n" + "=" * 50)
    print("Cross-process objective check (baseline vs session)")
    print("=" * 50)
    all_ok = True
    for b_rec, s_rec in zip(baseline["objective_records"], session["objective_records"]):
        if b_rec["label"] != s_rec["label"]:
            raise AssertionError("objective record label mismatch between processes")
        delta = abs(b_rec["cuopt_primal_objective"] - s_rec["cuopt_primal_objective"])
        ok = delta <= obj_atol
        all_ok = all_ok and ok
        print(
            f"  {b_rec['label']:>8}: delta={delta:.2e}  "
            f"baseline={b_rec['cuopt_primal_objective']:.10f}  "
            f"session={s_rec['cuopt_primal_objective']:.10f}  "
            f"{'OK' if ok else 'FAIL'}"
        )
    if not all_ok:
        raise AssertionError("cross-process objective verification failed")
    print("  Cross-process objective check: PASS")


def run_single_benchmark(mode: str) -> dict:
    """Run cold vs warm benchmark for one mode in the current process."""
    use_session = mode == "session"
    label = (
        "Session disabled (new handle each solve)"
        if mode == "baseline"
        else "Session enabled (reuse handle + symbolic cache)"
    )

    if os.environ.get("CUOPT_CACHE_PROFILE", "1").lower() not in ("0", "false", "no"):
        os.environ["CUOPT_CACHE_PROFILE"] = "1"

    k, scale = 50, 100
    if os.environ.get("CUOPT_PORTFOLIO_BENCHMARK_SMALL", "").lower() in ("1", "true", "yes"):
        k, scale = 5, 2

    print("=" * 70)
    print(f"Portfolio QP benchmark — mode={mode}")
    print("=" * 70)
    print(f"\nProblem settings: k={k}, scale={scale} -> n={k * scale} stocks, {k} factors")

    info = generate_portfolio_data(k=k, scale=scale, seed=42)
    same_values = os.environ.get("CUOPT_CACHE_SAME_VALUES", "").lower() in (
        "1",
        "true",
        "yes",
    )
    n_warm = int(os.environ.get("CUOPT_CACHE_BENCHMARK_WARM", "5"))
    obj_atol = float(os.environ.get("CUOPT_OBJ_VERIFY_ATOL", "1e-6"))
    formula_atol = float(os.environ.get("CUOPT_FORMULA_VERIFY_ATOL", "1e-4"))

    print("\nBuilding cuOpt Problem...")
    t_build = time.time()
    prob = build_cuopt_portfolio_problem(info)
    print(f"Structure build time: {time.time() - t_build:.2f} s")
    print(
        "\nMode: "
        + (
            "identical parameters on every solve"
            if same_values
            else "value-only parameter updates on warm solves"
        )
    )

    settings = _fresh_settings(use_session=use_session)
    bench = run_cache_benchmark(
        prob,
        info,
        settings,
        n_warm=n_warm,
        same_values=same_values,
        use_session=use_session,
    )
    _print_bench_summary(label, bench, n_warm)
    _assert_objectives(bench["objective_records"], formula_atol=formula_atol)
    print("  Objective check: PASS (Optimal; cuOpt vs formula within tolerance)")
    _assert_log_expectations(mode, bench)

    artifact = Path(os.environ.get("CUOPT_BENCHMARK_ARTIFACT", f"/tmp/cuopt_bench_{mode}.json"))
    _write_bench_artifact(artifact, mode, bench)
    print(f"  Wrote artifact: {artifact}")
    return bench


def main():
    parser = argparse.ArgumentParser(description="Portfolio QP session cache benchmark")
    parser.add_argument(
        "--mode",
        choices=("baseline", "session", "all"),
        default=os.environ.get("CUOPT_BENCHMARK_MODE", "all"),
        help="baseline=session off; session=session on (fresh process when invoked via --mode all)",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(1)

    script = Path(__file__).resolve()
    env = os.environ.copy()

    if args.mode == "all":
        print("=" * 70)
        print("Running two-process benchmark: baseline then session")
        print("=" * 70)
        baseline_artifact = Path("/tmp/cuopt_bench_baseline.json")
        session_artifact = Path("/tmp/cuopt_bench_session.json")
        env["CUOPT_BENCHMARK_ARTIFACT"] = str(baseline_artifact)
        print("\n>>> Process 1/2: baseline (session disabled)")
        rc = subprocess.run(
            ["stdbuf", "-eL", "-oL", sys.executable, str(script), "--mode", "baseline"],
            env=env,
            check=False,
        )
        if rc.returncode != 0:
            sys.exit(rc.returncode)

        env["CUOPT_BENCHMARK_ARTIFACT"] = str(session_artifact)
        print("\n>>> Process 2/2: session (session enabled, new process)")
        rc = subprocess.run(
            ["stdbuf", "-eL", "-oL", sys.executable, str(script), "--mode", "session"],
            env=env,
            check=False,
        )
        if rc.returncode != 0:
            sys.exit(rc.returncode)

        obj_atol = float(os.environ.get("CUOPT_OBJ_VERIFY_ATOL", "1e-6"))
        _compare_mode_objectives(baseline_artifact, session_artifact, obj_atol=obj_atol)

        baseline_bench = json.loads(baseline_artifact.read_text(encoding="utf-8"))
        session_bench = json.loads(session_artifact.read_text(encoding="utf-8"))
        delta = baseline_bench["warm_avg_ms"] - session_bench["warm_avg_ms"]
        print("\n" + "=" * 50)
        print("Timing summary (separate processes)")
        print("=" * 50)
        print(f"  Baseline warm avg:  {baseline_bench['warm_avg_ms']:.2f} ms")
        print(f"  Session warm avg:   {session_bench['warm_avg_ms']:.2f} ms")
        print(f"  Session savings:    {delta:.2f} ms")
        print("\nAll benchmarks and assertions PASSED.")
        return

    run_single_benchmark(args.mode)
    print("\nAll assertions PASSED.")


if __name__ == "__main__":
    main()
