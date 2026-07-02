# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for solver-session / symbolic-cache pytest and benchmarks."""

from __future__ import annotations

import io
import os
import re
import sys
from contextlib import contextmanager

import numpy as np

from cuopt.linear_programming import solver_settings
from cuopt.linear_programming.problem import (
    LinearExpression,
    MINIMIZE,
    Problem,
    QuadraticExpression,
)
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_AUGMENTED,
    CUOPT_METHOD,
    CUOPT_PRESOLVE,
)
from cuopt.linear_programming.solver_settings import SolverMethod

_REUSE_SYMBOLIC_LINE = re.compile(
    r"Barrier: reusing cuDSS symbolic analysis \(sparsity hash match\)"
)
_REBUILT_SYMBOLIC_LINE = re.compile(
    r"Barrier: rebuilt cuDSS symbolic analysis"
)
_STORE_AUGMENTED_LINE = re.compile(
    r"Barrier: stored augmented symbolic cache hash=0x[0-9a-f]+"
)
_STORE_ADAT_LINE = re.compile(
    r"Barrier: stored ADAT symbolic cache hash=0x[0-9a-f]+"
)
_STORE_HASH_LINE = re.compile(
    r"Barrier: stored (?:ADAT|augmented) symbolic cache hash=(0x[0-9a-f]+)"
)
_CLEAR_CACHE_LINE = re.compile(
    r"Barrier: hash match but numeric refresh failed \(CUDA\); clearing symbolic cache"
)
_LINEAR_SYSTEM_ADAT = re.compile(r"Linear system\s+:\s+ADAT")
_LINEAR_SYSTEM_AUGMENTED = re.compile(r"Linear system\s+:\s+augmented")
_CACHE_PROFILE_LINE = re.compile(
    r"^Cache profile: (C\d+) .+? ([0-9]+(?:\.[0-9]+)?)\s*$"
)


def count_log_matches(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def parse_cache_profile(text: str) -> dict[str, float]:
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


@contextmanager
def capture_solver_output():
    """Capture solver stdout and stderr (C++ logs use both)."""
    read_out, write_out = os.pipe()
    read_err, write_err = os.pipe()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    capture = io.StringIO()
    try:
        os.dup2(write_out, 1)
        os.dup2(write_err, 2)
        os.close(write_out)
        os.close(write_err)
        yield capture
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        with os.fdopen(read_out, "r", encoding="utf-8", errors="replace") as reader_out:
            out_text = reader_out.read()
        with os.fdopen(read_err, "r", encoding="utf-8", errors="replace") as reader_err:
            err_text = reader_err.read()
        text = out_text + err_text
        capture.write(text)
        capture.seek(0)
        if text:
            sys.stdout.write(out_text)
            sys.stdout.flush()
            sys.stderr.write(err_text)
            sys.stderr.flush()


def session_barrier_settings(
    *,
    session_enabled: bool = True,
    augmented: int = -1,
) -> solver_settings.SolverSettings:
    """Barrier settings with optional session and augmented/ADAT override."""
    ss = solver_settings.SolverSettings()
    ss.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)
    ss.set_parameter(CUOPT_PRESOLVE, 0)
    if augmented != -1:
        ss.set_parameter(CUOPT_AUGMENTED, augmented)
    if session_enabled:
        ss.set_session_enabled(True)
    return ss


def lp_problem_dims() -> tuple[int, int, int]:
    """Return (n_vars, n_rows, nnz_per_row) sized for meaningful symbolic work."""
    small = os.environ.get("CUOPT_SESSION_TEST_SMALL", "").lower() in ("1", "true", "yes")
    if small:
        return 300, 150, 6
    return 1200, 600, 8


def build_sparse_lp(
    *,
    seed: int = 42,
    n: int | None = None,
    m: int | None = None,
    nnz_per_row: int | None = None,
) -> tuple[Problem, list, np.ndarray]:
    """
    Random sparse LP (no quadratic term) suitable for the ADAT barrier path.

    Returns ``(problem, variables, objective_coefficients)``.
    """
    default_n, default_m, default_nnz = lp_problem_dims()
    n = n if n is not None else default_n
    m = m if m is not None else default_m
    nnz_per_row = nnz_per_row if nnz_per_row is not None else default_nnz

    rng = np.random.default_rng(seed)
    prob = Problem("sparse_lp_session")
    xs = [prob.addVariable(lb=0.0, name=f"x{i}") for i in range(n)]
    k = min(nnz_per_row, n)
    for j in range(m):
        cols = rng.choice(n, size=k, replace=False)
        coeffs = rng.uniform(0.5, 2.0, size=k)
        rhs = float(rng.uniform(50.0, 200.0))
        prob.addConstraint(
            LinearExpression([xs[i] for i in cols], coeffs.tolist(), 0.0) <= rhs
        )
    c = rng.uniform(1.0, 10.0, size=n)
    prob.setObjective(LinearExpression(xs, c.tolist(), 0.0), sense=MINIMIZE)
    return prob, xs, c


def perturb_lp_values(prob: Problem, xs: list, c: np.ndarray, seed: int) -> None:
    """Value-only update: objective coefficients and one constraint RHS."""
    rng = np.random.default_rng(seed)
    c[:] = c * (1.0 + 0.002 * rng.standard_normal(c.size))
    lin = LinearExpression(xs, c.tolist(), 0.0)
    prob.setObjective(lin, sense=MINIMIZE)
    if prob.constrs:
        prob.constrs[0].RHS = float(prob.constrs[0].RHS) * (
            1.0 + 0.001 * rng.standard_normal()
        )


def rewire_lp_row_sparsity(
    prob: Problem,
    xs: list,
    *,
    row_idx: int = 0,
    seed: int,
    nnz_per_row: int | None = None,
) -> None:
    """
    Change which variables appear in one constraint (same m, n; new A pattern).

    Does not add or remove variables or constraints — only the row's column indices.
    """
    rng = np.random.default_rng(seed)
    n = len(xs)
    constr = prob.constrs[row_idx]
    old_cols = set(constr.vindex_coeff_dict.keys())
    k = nnz_per_row if nnz_per_row is not None else max(len(old_cols), 1)
    k = min(k, n)
    for _ in range(20):
        new_cols = rng.choice(n, size=k, replace=False)
        if set(new_cols) != old_cols:
            break
    else:
        raise RuntimeError("could not pick a different sparsity pattern for constraint row")

    constr.vindex_coeff_dict.clear()
    for j in new_cols:
        constr.vindex_coeff_dict[int(j)] = float(rng.uniform(0.5, 2.0))
    constr.RHS = float(rng.uniform(50.0, 200.0))
    prob.solved = False
    prob.warmstart_data = None
    prob.model = None
    prob.constraint_csr_matrix = None
    prob._invalidate_index_to_var_cache()


def build_augmented_qp(
    *,
    seed: int = 11,
    n: int = 60,
    m: int = 30,
    nnz_per_row: int = 4,
) -> tuple[Problem, list, np.ndarray]:
    """Small QP with off-diagonal ``Q`` (augmented KKT path)."""
    rng = np.random.default_rng(seed)
    prob = Problem("augmented_qp_session")
    xs = [prob.addVariable(lb=0.0, ub=5.0, name=f"x{i}") for i in range(n)]
    k = min(nnz_per_row, n)
    for i in range(m):
        cols = rng.choice(n, size=k, replace=False)
        coeffs = rng.uniform(0.5, 2.0, size=k)
        rhs = float(rng.uniform(10.0, 50.0))
        prob.addConstraint(
            LinearExpression([xs[j] for j in cols], coeffs.tolist(), 0.0) <= rhs
        )
    c = rng.uniform(-5.0, -1.0, size=n)
    qv1, qv2, qc = [], [], []
    for i in range(n):
        qv1.append(xs[i])
        qv2.append(xs[i])
        qc.append(float(rng.uniform(0.5, 2.0)))
    for i in range(min(4, n - 1)):
        qv1.append(xs[i])
        qv2.append(xs[i + 1])
        qc.append(float(rng.uniform(0.1, 0.4)))
    prob.setObjective(
        QuadraticExpression(
            qvars1=qv1,
            qvars2=qv2,
            qcoefficients=qc,
            vars=[],
            coefficients=[],
            constant=0.0,
        )
        + LinearExpression(xs, c.tolist(), 0.0),
        sense=MINIMIZE,
    )
    return prob, xs, c


def perturb_qp_values(prob: Problem, xs: list, c: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    c[:] = c * (1.0 + 0.002 * rng.standard_normal(c.size))
    for i, var in enumerate(xs):
        var.setObjectiveCoefficient(float(c[i]))
    if prob.constrs:
        prob.constrs[0].RHS = float(prob.constrs[0].RHS) * (
            1.0 + 0.001 * rng.standard_normal()
        )
    prob.solved = False
    prob.warmstart_data = None


def solve_with_log(prob: Problem, settings, session=None):
    """Run ``prob.solve`` and return ``(solution, log_text, cache_profile)``."""
    with capture_solver_output() as capture:
        solution = prob.solve(settings, session=session)
    log_text = capture.getvalue()
    return solution, log_text, parse_cache_profile(log_text)


def assert_optimal(solution) -> None:
    assert solution.get_termination_reason() == "Optimal"


def assert_warm_symbolic_reuse(
    cold_log: str,
    warm_log: str,
    cold_profile: dict[str, float],
    warm_profile: dict[str, float],
    *,
    expect_adat: bool = False,
    expect_augmented: bool = False,
) -> None:
    """Cold run stores cache; warm run reuses symbolic factorization."""
    if expect_adat:
        assert count_log_matches(cold_log, _LINEAR_SYSTEM_ADAT) >= 1
        assert count_log_matches(cold_log, _STORE_ADAT_LINE) >= 1
    if expect_augmented:
        assert count_log_matches(cold_log, _LINEAR_SYSTEM_AUGMENTED) >= 1
        assert count_log_matches(cold_log, _STORE_AUGMENTED_LINE) >= 1

    assert count_log_matches(cold_log, _REUSE_SYMBOLIC_LINE) == 0

    warm_reuse = count_log_matches(warm_log, _REUSE_SYMBOLIC_LINE)
    c07_c = cold_profile.get("C07", 0.0)
    c07_w = warm_profile.get("C07", 0.0)

    if warm_reuse >= 1:
        assert count_log_matches(warm_log, _REBUILT_SYMBOLIC_LINE) == 0
    else:
        # Logs may be buffered when stderr is piped; fall back to C07 timing.
        assert c07_c > 0.0, "expected non-zero C07 on cold symbolic factorization"
        assert c07_w <= max(1.0, 0.05 * c07_c), (
            f"expected warm C07 near zero with reuse (cold={c07_c:.2f} ms, warm={c07_w:.2f} ms)"
        )


def stored_sparsity_hashes(text: str) -> list[str]:
    return _STORE_HASH_LINE.findall(text)


def assert_full_symbolic_reanalyze(
    log_text: str,
    profile: dict[str, float],
    *,
    cold_c07: float | None = None,
    allow_cache_clear: bool = False,
) -> None:
    """
    Warm solve after sparsity change must not reuse cached symbolic analysis.

    When ``allow_cache_clear`` is true, a false hash match that clears the cache
    (numeric refresh CUDA failure) is accepted; caller should retry the solve.
    """
    if count_log_matches(log_text, _REUSE_SYMBOLIC_LINE) > 0:
        raise AssertionError("unexpected symbolic reuse log after sparsity change")
    if allow_cache_clear and count_log_matches(log_text, _CLEAR_CACHE_LINE) >= 1:
        return
    c07 = profile.get("C07", 0.0)
    rebuilt = count_log_matches(log_text, _REBUILT_SYMBOLIC_LINE) >= 1
    if cold_c07 is not None and cold_c07 > 5.0:
        assert rebuilt or c07 >= 0.1 * cold_c07, (
            f"expected full re-analyze (rebuilt log or C07; got C07={c07:.2f} ms, "
            f"cold ref={cold_c07:.2f} ms, rebuilt={rebuilt})"
        )
    else:
        assert c07 > 0.0 or rebuilt
