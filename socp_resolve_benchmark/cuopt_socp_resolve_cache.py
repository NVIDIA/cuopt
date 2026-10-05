#!/usr/bin/env python3
"""Cache reuse on the portfolio SOCP: cold solves vs update_linear_objective.

Companion to adat_vs_aug_lin_obj.py, which measures the same thing on the QP
form of this portfolio. Here the risk cap is a quadratic *constraint*, so cuOpt
expands it to a second-order cone and the reuse path exercised is the one added
for SOCP. The model and the 20-step mu sequence are those of
cuopt_socp_resolve.py, which solves every step cold.

Per config the series are:

  baseline  K cold solves, a fresh DataModel per mu, run twice
  cache     t=0 stores the cache, t=1..K-1 update_linear_objective and reuse
  control   K cold solves pinned to SedumiMu, also twice, and only when reuse
            overrides the requested initial point

Each reference is run twice because this solver does not reproduce itself
exactly here: two identical cold series disagree on a few steps, and at the
default step scale repeating one cold solve gives iteration counts anywhere
in 24..31. So "cold vs cold" is measured first and becomes the band that
"reuse vs cold" has to fall inside. Fixed thresholds would either pass
everything or fail on the solver's own noise. barrier_step_scale=0.99 tightens
the band considerably, which is why the tuned config measures more sharply.

Iterations are compared against the control rather than baseline. On a cone
model reuse forces SedumiMu, because reset_iterate_state leaves the previous
solve's converged NT scaling in the cone block and every other initial point
would factorize it. The tuned config asks for DualLeastSquares, so its cold
and warm counts differ by design, and the control is what separates that from
a regression. Objectives are compared against baseline directly.

Timings are wall clock and assume an otherwise idle GPU. Check nvidia-smi
before trusting the improvement table.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import re
import statistics as st
import tempfile
import time

import numpy as np
import scipy.sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.path.join(SCRIPT_DIR, "data"))
ap.add_argument("--K", type=int, default=20)
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--ir", type=int, default=1)
ap.add_argument("--augmented", type=int, default=1)
ap.add_argument("--pool-gib", type=float, default=1.0)
ap.add_argument("--configs", default="default,tuned")
ap.add_argument("--obj-atol", type=float, default=1e-9, help="floor for the objective band")
ap.add_argument(
    "--cudss-deterministic",
    type=int,
    default=0,
    help="cuDSS deterministic mode; with the ALG2 SpMV fix this makes runs bit-identical",
)
ap.add_argument(
    "--csr-ir-matvec",
    type=int,
    default=0,
    help="do the refinement matvec as one SpMV over the augmented CSR instead of matrix-free; "
    "needs --augmented 1 and --ir non-zero to have any effect",
)
ap.add_argument(
    "--initial-point",
    type=int,
    default=None,
    help="pin barrier_dual_initial_point for every series (2 = SedumiMu)",
)
ap.add_argument(
    "--step-scale", type=float, default=None, help="override barrier_step_scale for every series"
)
ap.add_argument("--save-logs", default="", help="directory to write each solve's log into")
ap.add_argument(
    "--repeat",
    type=int,
    default=0,
    help="solve one instance this many times cold and report whether the runs agree. "
    "Measures the solver's reproducibility rather than reuse",
)
ap.add_argument(
    "--repeat-step",
    type=int,
    default=1,
    help="which mu of the sequence --repeat solves; 1 matches the earlier nondeterminism probes",
)
ap.add_argument(
    "--cold-only",
    action="store_true",
    help="run just the K cold solves and report their outcomes. Skips the reuse series, "
    "which matters for configurations that break down: a hard failure at t=0 leaves no "
    "cache and the next update_linear_objective aborts the whole run",
)
ap.add_argument(
    "--nd-nlevels",
    type=int,
    default=-1,
    help="cudss_hyper_nd_nlevels: METIS nested-dissection depth, -1 leaves it to cuDSS",
)
ap.add_argument(
    "--set",
    action="append",
    default=[],
    metavar="NAME=VALUE",
    help="any other solver parameter, repeatable; ints/floats/bools are parsed from the text",
)
ap.add_argument(
    "--bound-free-vars",
    type=int,
    default=0,
    help="barrier_presolve_bound_free_variables: -1 automatic, 0 disabled, 1 enabled. "
    "Reuse requires 0, and cuOpt already resolves -1 to 0 under sequence_solve, so 0 is "
    "what makes the cold baseline presolve the same way the warm series does",
)
a = ap.parse_args()

if a.pool_gib > 0:
    import rmm

    rmm.mr.set_current_device_resource(
        rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
            initial_pool_size=int(a.pool_gib * (1 << 30)),
        )
    )

from cuopt.linear_programming import (  # noqa: E402
    DataModel,
    Solve,
    SolverMethod,
    SolverSettings,
)
from cuopt.linear_programming.solver import solver_parameters as PAR  # noqa: E402

# barrier_dual_initial_point
DUAL_LEAST_SQUARES = 1
SEDUMI_MU = 2

_OPT = re.compile(r"Optimal solution found in (\d+) iterations and ([\d.]+)s")
# A soft breakdown reports its own iteration count and time; a hard one never gets that far.
_SUBOPT = re.compile(r"Suboptimal solution found in (\d+) iterations and ([\d.]+) seconds")
_HARD = re.compile(r"Search direction computation failed")
_CACHE_REUSE = re.compile(r"Barrier: reusing cache \(skip convert/presolve/scaling\)")
_OVERRIDE = re.compile(r"overrides the requested initial point")
_CONES = re.compile(r"Second-order cones\s+:\s+(\d+)")

D = np.load(os.path.join(a.data_dir, "portfolio.npz"))
F, d, mu = D["F"], D["d"], D["mu"]
n, k, cap = int(D["n"]), int(D["k"]), float(D["cap"])
sigma2 = float(np.load(os.path.join(a.data_dir, "portfolio_socp.npz"))["sigma2"])
SEQ = np.load(os.path.join(a.data_dir, "resolve_seq.npz"))["mu_seq"]
K = min(a.K, SEQ.shape[0])
INF = float("inf")

budget = sp.hstack([sp.csr_matrix(np.ones((1, n))), sp.csr_matrix((1, k))])
couple = sp.hstack([sp.csr_matrix(F.T), -sp.eye(k)])
A = sp.vstack([budget, couple], format="csr")
b = np.concatenate([[1.0], np.zeros(k)])
vlb = np.concatenate([np.zeros(n), np.full(k, -INF)])
vub = np.concatenate([np.full(n, cap), np.full(k, INF)])
Qd = np.concatenate([d, np.ones(k)]).astype(np.float64)
Qi = np.arange(n + k, dtype=np.int32)


def cof(muv):
    """Objective for a given expected-return vector; the k factor columns carry none."""
    return np.concatenate([-muv, np.zeros(k)]).astype(np.float64)


def build(c):
    dm = DataModel()
    dm.set_csr_constraint_matrix(
        A.data.astype(np.float64), A.indices.astype(np.int32), A.indptr.astype(np.int32)
    )
    dm.set_constraint_lower_bounds(b)
    dm.set_constraint_upper_bounds(b)
    dm.set_objective_coefficients(c)
    dm.set_variable_lower_bounds(vlb)
    dm.set_variable_upper_bounds(vub)
    dm.add_quadratic_constraint("risk", None, None, sigma2, Qd, Qi, Qi, "L")
    return dm


def settings(*, tuned, sequence_solve, initial_point=None):
    s = SolverSettings()
    s.set_parameter(PAR.CUOPT_METHOD, SolverMethod.Barrier)
    s.set_parameter(PAR.CUOPT_AUGMENTED, a.augmented)
    try:
        s.set_parameter(PAR.CUOPT_CROSSOVER, False)
    except Exception:
        pass
    s.set_parameter(PAR.CUOPT_BARRIER_ITERATIVE_REFINEMENT, a.ir)
    # Too new to be in the installed Python bindings' PAR constants, so name it directly.
    s.set_parameter("barrier_csr_ir_matvec", bool(a.csr_ir_matvec))
    s.set_parameter(PAR.CUOPT_BARRIER_PRESOLVE_BOUND_FREE_VARIABLES, a.bound_free_vars)
    s.set_parameter(PAR.CUOPT_SEQUENCE_SOLVE, sequence_solve)
    s.set_parameter(PAR.CUOPT_CUDSS_DETERMINISTIC, bool(a.cudss_deterministic))
    # Not in this tree's bindings. Default -1 means leave nested dissection to cuDSS.
    if hasattr(PAR, "CUOPT_CUDSS_HYPER_ND_NLEVELS"):
        s.set_parameter(PAR.CUOPT_CUDSS_HYPER_ND_NLEVELS, a.nd_nlevels)
    s.set_optimality_tolerance(a.tol)
    if tuned:
        s.set_parameter(PAR.CUOPT_BARRIER_DUAL_INITIAL_POINT, DUAL_LEAST_SQUARES)
        s.set_parameter(PAR.CUOPT_BARRIER_STEP_SCALE, 0.99)
    # A pinned control still wins over the CLI, so the control series stays a control.
    if initial_point is None:
        initial_point = a.initial_point
    if initial_point is not None:
        s.set_parameter(PAR.CUOPT_BARRIER_DUAL_INITIAL_POINT, initial_point)
    if a.step_scale is not None:
        s.set_parameter(PAR.CUOPT_BARRIER_STEP_SCALE, a.step_scale)
    for item in a.set:
        name, _, text = item.partition("=")
        s.set_parameter(name, _coerce(text))
    return s


def _coerce(text):
    if text in ("True", "False"):
        return text == "True"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


_libc = ctypes.CDLL(None)


def solve_capture(dm, s, log_path=None):
    """Solve with fds 1 and 2 redirected; the C++ log bypasses sys.stdout."""
    tf = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log")
    old_out, old_err = os.dup(1), os.dup(2)
    os.dup2(tf.fileno(), 1)
    os.dup2(tf.fileno(), 2)
    try:
        t0 = time.perf_counter()
        sol = Solve(dm, s)
        wall = (time.perf_counter() - t0) * 1e3
    finally:
        # Without this the tail of the C++ log is still sitting in libc's
        # buffer when the fds are restored, and lands in the terminal instead
        # of the file. It shows up as the occasional run with no
        # "Optimal solution found" line.
        _libc.fflush(None)
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        tf.flush()
    log = open(tf.name).read()
    os.unlink(tf.name)
    if log_path and a.save_logs:
        full = os.path.join(a.save_logs, log_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(log)
    return sol, log, wall


def parse(sol, log, wall):
    m = _OPT.search(log)
    sub = _SUBOPT.search(log)
    suboptimal = "Suboptimal solution found" in log
    hard = bool(_HARD.search(log))
    return dict(
        # Two ways a solve breaks down: soft rolls back to the last good iterate and still
        # reports its iteration count, hard produces nan residuals and reports nothing.
        mode="hard" if hard else ("soft" if suboptimal else ""),
        # iters/barrier_ms below stay None on any failure so they cannot contaminate the
        # reuse comparisons. These two are the same numbers with the soft line as a fallback,
        # for tables that want a value for every solve.
        any_iters=int(m.group(1)) if m else (int(sub.group(1)) if sub else None),
        any_ms=float(m.group(2)) * 1e3 if m else (float(sub.group(2)) * 1e3 if sub else None),
        # What the barrier itself concluded. get_termination_reason reports Optimal
        # even for the runs the log calls suboptimal, so the log is the honest source.
        outcome="numerical_error" if suboptimal else ("optimal" if m else "unknown"),
        wall_ms=wall,
        # Absent whenever the run did not end on the optimal-solution line.
        barrier_ms=float(m.group(2)) * 1e3 if m else None,
        # None when the run never printed the optimal-solution line, so it is
        # left out of the comparisons instead of entering them as a zero.
        iters=int(m.group(1)) if m else None,
        status=str(sol.get_termination_reason()),
        obj=float(sol.get_primal_objective()),
        cones=int(_CONES.search(log).group(1)) if _CONES.search(log) else 0,
        reuse=bool(_CACHE_REUSE.search(log)),
        override=bool(_OVERRIDE.search(log)),
    )


def run_cold(tuned, initial_point=None, tag="cold"):
    s = settings(tuned=tuned, sequence_solve=False, initial_point=initial_point)
    return [
        parse(*solve_capture(build(cof(SEQ[t])), s, f"{tag}/t{t:02d}.log")) for t in range(K)
    ]


def run_cache(tuned, tag="warm"):
    """t=0 stores the cache, the rest reuse it. Returns fewer than K rows if the cache is lost.

    A hard failure returns a non-optimal status and solve.cpp then clears the whole
    cache, so the next update_linear_objective finds no transform. Soft breakdowns
    report OPTIMAL and keep it. Truncate the series instead of aborting the run, so
    the steps that did reuse still get measured.
    """
    s = settings(tuned=tuned, sequence_solve=True)
    dm = build(cof(SEQ[0]))
    rows = [parse(*solve_capture(dm, s, f"{tag}/t00.log"))]
    for t in range(1, K):
        try:
            dm.update_linear_objective(cof(SEQ[t]))
        except Exception as exc:
            print(f"  cache lost after t={t - 1}, series stops there: {exc}")
            break
        rows.append(parse(*solve_capture(dm, s, f"{tag}/t{t:02d}.log")))
    return rows


def mean(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return st.mean(vals) if vals else math.nan


def max_gap(xs, ys, key):
    """Worst per-step gap, skipping steps where either side has no value."""
    gaps = [
        abs(x[key] - y[key])
        for x, y in zip(xs, ys)
        if x[key] is not None and y[key] is not None
    ]
    return max(gaps) if gaps else math.nan


def bad_solves(series):
    return [
        (name, t, r["status"])
        for name, rows in series.items()
        for t, r in enumerate(rows)
        if r["status"] != "Optimal" or r["iters"] is None
    ]


def cell(v, width=8):
    return f"{'-':>{width}}" if v is None else f"{v:>{width}}"


def summarize_repeat(config, tuned):
    """Solve one instance N times cold. Any disagreement is the solver failing to reproduce itself."""
    s = settings(tuned=tuned, sequence_solve=False)
    muv = SEQ[a.repeat_step]
    rows = [
        parse(*solve_capture(build(cof(muv)), s, f"{config}/repeat/r{i:02d}.log"))
        for i in range(a.repeat)
    ]
    first = rows[0]
    print(f"\n=== {config}: {a.repeat} cold solves of SEQ[{a.repeat_step}] ===")
    hdr = (
        f"{'run':>4} | {'it':>3} {'barr_ms':>8} {'mode':>5} | {'objective':>22}"
        f" | {'d_it':>4} {'d_obj':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows):
        ms = "-" if r["any_ms"] is None else f"{r['any_ms']:.1f}"
        d_it = (
            "-"
            if r["any_iters"] is None or first["any_iters"] is None
            else f"{r['any_iters'] - first['any_iters']:+d}"
        )
        print(
            f"{i:>4} | {cell(r['any_iters'], 3)} {ms:>8} {r['mode'] or 'ok':>5}"
            f" | {r['obj']:>22.17g} | {d_it:>4} {r['obj'] - first['obj']:>+10.2e}"
        )

    its = sorted({r["any_iters"] for r in rows if r["any_iters"] is not None})
    objs = [r["obj"] for r in rows]
    # Exact equality, not a tolerance: the question is whether the runs are bit-identical.
    same_obj = len(set(objs)) == 1
    spread = max(objs) - min(objs)
    soft = sum(r["mode"] == "soft" for r in rows)
    hard = sum(r["mode"] == "hard" for r in rows)
    print(f"  iterations: {len(its)} distinct value(s) {its}")
    print(
        f"  objectives: {len(set(objs))} distinct value(s), spread {spread:.2e}"
        f" ({'bit-identical' if same_obj else 'NOT identical'})"
    )
    print(f"  outcomes: {len(rows) - soft - hard} optimal, {soft} soft, {hard} hard")
    return dict(
        config=config,
        ok=same_obj and len(its) == 1 and (soft + hard) == 0,
        n=len(rows),
        soft=soft,
        hard=hard,
        n_it=len(its),
        its=its,
        n_obj=len(set(objs)),
        spread=spread,
        iters=mean(rows, "any_iters"),
        barrier_ms=mean(rows, "any_ms"),
    )


def summarize_cold(config, tuned):
    rows = run_cold(tuned, tag=f"{config}/cold")
    print(f"\n=== {config} (cold only) ===")
    hdr = f"{'t':>4} | {'it':>3} {'barr_ms':>8} {'mode':>5} {'outcome':>15} {'objective':>18}"
    print(hdr)
    print("-" * len(hdr))
    for t, r in enumerate(rows):
        ms = "-" if r["any_ms"] is None else f"{r['any_ms']:.1f}"
        print(
            f"{t:>4} | {cell(r['any_iters'], 3)} {ms:>8} {r['mode'] or 'ok':>5}"
            f" {r['outcome']:>15} {r['obj']:>18.12f}"
        )
    soft = sum(r["mode"] == "soft" for r in rows)
    hard = sum(r["mode"] == "hard" for r in rows)
    print(f"  {len(rows) - soft - hard}/{len(rows)} optimal, {soft} soft, {hard} hard")
    return dict(
        config=config,
        ok=(soft + hard) == 0,
        n=len(rows),
        soft=soft,
        hard=hard,
        iters=mean(rows, "any_iters"),
        barrier_ms=mean(rows, "any_ms"),
        wall_ms=mean(rows, "wall_ms"),
        ok_iters=mean(rows, "iters"),
        ok_barrier_ms=mean(rows, "barrier_ms"),
    )


def summarize(config, tuned):
    base1 = run_cold(tuned, tag=f"{config}/cold")
    base2 = run_cold(tuned, tag=f"{config}/cold_repeat")
    cache = run_cache(tuned, tag=f"{config}/warm")
    # t=0 builds the cache, so only t>=1 exercises reuse. The cold references are cut to
    # the same steps, since a lost cache leaves the warm series short and the means would
    # otherwise average different subsequences.
    warm, base_warm = cache[1:], base1[1 : len(cache)]
    truncated = len(cache) < K

    overrode = any(r["override"] for r in warm)
    if overrode:
        ctl1 = run_cold(tuned, SEDUMI_MU, tag=f"{config}/control")
        ctl2 = run_cold(tuned, SEDUMI_MU, tag=f"{config}/control_repeat")
    else:
        ctl1, ctl2 = base1, base2
    ctl_warm = ctl1[1 : len(cache)]

    # What the solver does against itself, and what reuse does against it.
    # Reuse also keeps the scaling computed at t=0, so it can land anywhere
    # inside the requested optimality tolerance even when cold solves happen
    # to agree more closely than that; hence the tolerance term.
    cold_gap = max_gap(base1, base2, "obj")
    tol_term = 10.0 * a.tol * abs(base1[0]["obj"])
    obj_band = max(a.obj_atol, cold_gap, tol_term)
    iter_band = max_gap(ctl1, ctl2, "iters")
    obj_dev = max_gap(cache, base1, "obj")
    iter_dev = max_gap(warm, ctl_warm, "iters")

    print(f"\n=== {config} ===")
    print(
        f"  cold vs cold: objectives differ by up to {cold_gap:.2e}, control "
        f"iterations by up to {iter_band}; 10x optimality tol is {tol_term:.2e}"
    )
    hdr = (
        f"{'t':>4} | {'it':>3} {'barr_ms':>8} {'outcome':>15} {'objective':>18}"
        f" | {'it':>3} {'barr_ms':>8} {'outcome':>15} {'objective':>18} | {'delta':>10}"
    )
    print(f"{'':>4} | {'--- COLD ---':^47} | {'--- WARM (reuse) ---':^47} |")
    print(hdr)
    print("-" * len(hdr))
    for t in range(K):
        mark = "*" if t == 0 else " "  # t=0 stores the cache, it is not a reuse
        c = base1[t]
        cb = "-" if c["barrier_ms"] is None else f"{c['barrier_ms']:.1f}"
        left = (
            f"{t:>3}{mark} | {cell(c['iters'], 3)} {cb:>8} {c['outcome']:>15} {c['obj']:>18.12f}"
        )
        if t >= len(cache):
            print(f"{left} | {'cache lost, no reuse to measure':^47} | {'-':>10}")
            continue
        w = cache[t]
        wb = "-" if w["barrier_ms"] is None else f"{w['barrier_ms']:.1f}"
        print(
            f"{left} | {cell(w['iters'], 3)} {wb:>8} {w['outcome']:>15} {w['obj']:>18.12f}"
            f" | {w['obj'] - c['obj']:>+10.2e}"
        )

    series = {"baseline": base1, "baseline2": base2, "cache": cache}
    if overrode:
        series.update({"control": ctl1, "control2": ctl2})
    bad = bad_solves(series)

    obj_ok = obj_dev <= obj_band
    iters_ok = iter_dev <= max(1, iter_band)
    reuse_ok = all(r["reuse"] for r in warm)
    status_ok = not bad
    exact = sum(
        w["iters"] == c["iters"]
        for w, c in zip(warm, ctl_warm)
        if w["iters"] is not None and c["iters"] is not None
    )

    print(f"  objectives, reuse vs cold: {'OK' if obj_ok else 'FAIL'}  "
          f"{obj_dev:.2e} against a cold-vs-cold band of {obj_band:.2e}")
    print(f"  iterations, reuse vs control: {'OK' if iters_ok else 'FAIL'}  "
          f"worst step off by {iter_dev} against a band of {max(1, iter_band)} "
          f"(exact on {exact}/{len(warm)})")
    print(f"  cache reused on every warm solve: {'OK' if reuse_ok else 'FAIL'}  "
          f"({sum(r['reuse'] for r in warm)}/{len(warm)})")
    print(f"  all solves Optimal: {'OK' if status_ok else 'FAIL'}")
    if truncated:
        print(
            f"  reuse series ran {len(cache)}/{K} steps: a hard failure clears the cache, so "
            "every later step is unreachable"
        )
    for name, t, status in bad:
        print(f"    {name} t={t}: {status}")
    if overrode:
        print(
            "  reuse overrode the requested initial point with SedumiMu, so warm "
            f"iterations track the control ({mean(ctl_warm, 'iters'):.1f}) rather "
            f"than baseline ({mean(base_warm, 'iters'):.1f})."
        )

    return dict(
        config=config,
        ok=obj_ok and iters_ok and reuse_ok and status_ok and not truncated,
        base_wall=mean(base_warm, "wall_ms"),
        warm_wall=mean(warm, "wall_ms"),
        base_barrier=mean(base_warm, "barrier_ms"),
        warm_barrier=mean(warm, "barrier_ms"),
        base_iters=mean(base_warm, "iters"),
        warm_iters=mean(warm, "iters"),
        ctl_iters=mean(ctl_warm, "iters"),
        obj_dev=obj_dev,
        obj_band=obj_band,
        n_warm=len(warm),
    )


def reuse_table(rows):
    print(f"\n=== reuse improvement (mean over {rows[0]['n_warm']} warm solves) ===")
    hdr = (
        f"{'config':8s} {'cold_wall':>10s} {'warm_wall':>10s} {'save':>9s} "
        f"{'save%':>7s}  {'cold_barr':>10s} {'warm_barr':>10s}  {'cold_it':>8s} "
        f"{'warm_it':>8s} {'ctl_it':>7s}  {'obj_dev':>9s} {'obj_band':>9s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        dw = r["base_wall"] - r["warm_wall"]
        print(
            f"{r['config']:8s} {r['base_wall']:10.1f} {r['warm_wall']:10.1f} "
            f"{dw:+9.1f} {100 * dw / r['base_wall']:6.1f}%  {r['base_barrier']:10.1f} "
            f"{r['warm_barrier']:10.1f}  {r['base_iters']:8.1f} "
            f"{r['warm_iters']:8.1f} {r['ctl_iters']:7.1f}  "
            f"{r['obj_dev']:9.1e} {r['obj_band']:9.1e}"
        )
    print(
        "\nwall = Solve() wall clock, the honest reuse metric: convert/presolve/"
        "scaling\nare what reuse skips and they sit outside the barrier timer."
        "\nbarr = barrier's own timer.  ctl_it = cold solve pinned to the initial"
        "\npoint reuse uses.  obj_dev = worst reuse-vs-cold objective gap,"
        "\nobj_band = worst cold-vs-cold gap, the noise it has to beat."
    )


def main():
    print(
        f"cuOpt SOCP cache reuse  n={n} k={k} sigma^2={sigma2:.6e}  K={K}  "
        f"tol={a.tol:.0e} IR={a.ir} augmented={a.augmented} pool={a.pool_gib} GiB"
    )
    # Discarded: the first solve of the process pays CUDA and cuDSS init.
    solve_capture(build(cof(mu)), settings(tuned=False, sequence_solve=False))

    wanted = [c.strip() for c in a.configs.split(",") if c.strip()]
    if a.repeat:
        rows = [
            summarize_repeat(c, tuned=(c == "tuned")) for c in ("default", "tuned") if c in wanted
        ]
        return 0 if all(r["ok"] for r in rows) else 1
    if a.cold_only:
        rows = [summarize_cold(c, tuned=(c == "tuned")) for c in ("default", "tuned") if c in wanted]
        return 0 if all(r["ok"] for r in rows) else 1
    rows = [summarize(c, tuned=(c == "tuned")) for c in ("default", "tuned") if c in wanted]
    reuse_table(rows)
    ok = all(r["ok"] for r in rows)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
