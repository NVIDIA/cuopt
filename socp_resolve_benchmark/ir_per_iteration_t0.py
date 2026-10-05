#!/usr/bin/env python3
"""Cost of one barrier iteration on a cached re-solve of the portfolio SOCP, with IR on and off.

Pins the problem to mu step t=0 and measures the same instance every time. The
first solve of a series stores the cache; the re-solves that follow call
update_linear_objective with the identical objective, so every timed solve is a
reuse of the same problem rather than a new one.

Per-iteration time comes from the Elapsed column of the barrier log, not from
dividing the total: the total includes setup and the final residual pass, which
on a 25-iteration solve is a few percent. (last - first) / (n - 1) over the
iteration lines is the steady-state cost and is insensitive to the 1 ms
quantization of that column.

Iterative refinement is the thing being priced. IR=0 is Off, 1 is GMRES (the
default), 2 is FixedPoint; barrier_csr_ir_matvec swaps the matrix-free
refinement matvec for one SpMV over the augmented CSR.
"""

from __future__ import annotations

import argparse
import ctypes
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
ap.add_argument("--step", type=int, default=0, help="which mu of the sequence to pin to")
ap.add_argument("--repeats", type=int, default=5, help="cached re-solves timed per cell")
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--pool-gib", type=float, default=1.0)
ap.add_argument("--out", default="", help="also write the table here")
ap.add_argument(
    "--build-retries",
    type=int,
    default=8,
    help="attempts at the cache-building solve before a cell is abandoned; IR off breaks "
    "down often enough at step 0.99 that one attempt frequently is not enough",
)
ap.add_argument(
    "--ref-obj",
    type=float,
    default=-0.098563711172,
    help="reference objective for the deviation column; the default is ClarabelGPU on t=0 "
    "at the same 1e-8 tolerance, from logs/clarabel_headtohead.log",
)
a = ap.parse_args()

if a.pool_gib > 0:
    import rmm

    rmm.mr.set_current_device_resource(
        rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(), initial_pool_size=int(a.pool_gib * (1 << 30))
        )
    )

from cuopt.linear_programming import (  # noqa: E402
    DataModel,
    Solve,
    SolverMethod,
    SolverSettings,
)
from cuopt.linear_programming.solver import solver_parameters as PAR  # noqa: E402

D = np.load(os.path.join(a.data_dir, "portfolio.npz"))
F, d = D["F"], D["d"]
n, k, cap = int(D["n"]), int(D["k"]), float(D["cap"])
sigma2 = float(np.load(os.path.join(a.data_dir, "portfolio_socp.npz"))["sigma2"])
SEQ = np.load(os.path.join(a.data_dir, "resolve_seq.npz"))["mu_seq"]
INF = float("inf")

budget = sp.hstack([sp.csr_matrix(np.ones((1, n))), sp.csr_matrix((1, k))])
couple = sp.hstack([sp.csr_matrix(F.T), -sp.eye(k)])
A = sp.vstack([budget, couple], format="csr")
b = np.concatenate([[1.0], np.zeros(k)])
vlb = np.concatenate([np.zeros(n), np.full(k, -INF)])
vub = np.concatenate([np.full(n, cap), np.full(k, INF)])
Qd = np.concatenate([d, np.ones(k)]).astype(np.float64)
Qi = np.arange(n + k, dtype=np.int32)
COF = np.concatenate([-SEQ[a.step], np.zeros(k)]).astype(np.float64)


def build():
    dm = DataModel()
    dm.set_csr_constraint_matrix(
        A.data.astype(np.float64), A.indices.astype(np.int32), A.indptr.astype(np.int32)
    )
    dm.set_constraint_lower_bounds(b)
    dm.set_constraint_upper_bounds(b)
    dm.set_objective_coefficients(COF)
    dm.set_variable_lower_bounds(vlb)
    dm.set_variable_upper_bounds(vub)
    dm.add_quadratic_constraint("risk", None, None, sigma2, Qd, Qi, Qi, "L")
    return dm


def settings(ir, csr, step_scale):
    s = SolverSettings()
    s.set_parameter(PAR.CUOPT_METHOD, SolverMethod.Barrier)
    s.set_parameter(PAR.CUOPT_AUGMENTED, 1)
    s.set_parameter(PAR.CUOPT_CROSSOVER, False)
    s.set_parameter(PAR.CUOPT_BARRIER_ITERATIVE_REFINEMENT, ir)
    s.set_parameter("barrier_csr_ir_matvec", bool(csr))
    s.set_parameter(PAR.CUOPT_BARRIER_PRESOLVE_BOUND_FREE_VARIABLES, 0)
    s.set_parameter(PAR.CUOPT_SEQUENCE_SOLVE, True)
    s.set_parameter(PAR.CUOPT_BARRIER_STEP_SCALE, step_scale)
    s.set_optimality_tolerance(a.tol)
    return s


_libc = ctypes.CDLL(None)
# Iteration lines: index, primal, dual, three residuals, elapsed seconds.
_ITER = re.compile(
    r"^\s*(\d+)\s+-?\d\.\d+e[+-]\d+\s+-?\d\.\d+e[+-]\d+\s+\S+\s+\S+\s+\S+\s+([\d.]+)\s*$", re.M
)
_OPT = re.compile(r"Optimal solution found in (\d+) iterations and ([\d.]+)s")
_SUB = re.compile(r"Suboptimal solution found in (\d+) iterations and ([\d.]+) seconds")
_HARD = re.compile(r"Search direction computation failed")
_REUSE = re.compile(r"Barrier: reusing cache \(skip convert/presolve/scaling\)")


def solve_capture(dm, s):
    tf = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log")
    old_out, old_err = os.dup(1), os.dup(2)
    os.dup2(tf.fileno(), 1)
    os.dup2(tf.fileno(), 2)
    try:
        t0 = time.perf_counter()
        sol = Solve(dm, s)
        wall = (time.perf_counter() - t0) * 1e3
    finally:
        _libc.fflush(None)
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        tf.flush()
    log = open(tf.name).read()
    os.unlink(tf.name)
    return sol, log, wall


def parse(sol, log, wall):
    m, sub = _OPT.search(log), _SUB.search(log)
    it = _ITER.findall(log)
    # Steady state only: the first line carries setup, the last the exit test.
    ms_per_it = None
    if len(it) >= 3:
        t_first, t_last = float(it[0][1]), float(it[-1][1])
        ms_per_it = (t_last - t_first) * 1e3 / (len(it) - 1)
    return dict(
        iters=int(m.group(1)) if m else (int(sub.group(1)) if sub else None),
        barrier_ms=float(m.group(2)) * 1e3 if m else (float(sub.group(2)) * 1e3 if sub else None),
        wall_ms=wall,
        ms_per_it=ms_per_it,
        n_lines=len(it),
        mode="hard" if _HARD.search(log) else ("soft" if sub else "ok"),
        obj=float(sol.get_primal_objective()),
        reuse=bool(_REUSE.search(log)),
    )


def run_cell(ir, csr, step_scale):
    """One cache build, then --repeats re-solves of the identical problem.

    A hard failure on the cache build stores no cache, and the update then
    reports the absence of a transform rather than the failure that caused it.
    Retry the build a few times before giving up on the cell.
    """
    s = settings(ir, csr, step_scale)
    for _ in range(a.build_retries):
        dm = build()
        first = parse(*solve_capture(dm, s))
        rows = []
        try:
            for _ in range(a.repeats):
                dm.update_linear_objective(COF)
                rows.append(parse(*solve_capture(dm, s)))
            return first, rows
        except Exception:
            continue
    return first, []


def med(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return st.median(vals) if vals else float("nan")


CELLS = [
    ("step 0.9", 0.9, 0, 0),
    ("step 0.9", 0.9, 1, 0),
    ("step 0.9", 0.9, 1, 1),
    ("step 0.9", 0.9, 2, 0),
    ("step 0.99", 0.99, 0, 0),
    ("step 0.99", 0.99, 1, 0),
    ("step 0.99", 0.99, 1, 1),
    ("step 0.99", 0.99, 2, 0),
]
IR_NAME = {0: "off", 1: "GMRES", 2: "FixedPt"}


def main():
    out = []

    def emit(line=""):
        print(line)
        out.append(line)

    emit(
        f"Cached re-solve of mu step t={a.step}: {a.repeats} re-solves per cell, one problem "
        f"throughout.\nn={n} k={k}, expanded to 5103 constraints x 10102 variables. tol={a.tol:.0e},"
        f" augmented=1,\nbound_free_vars=0, determinism off. Reuse pins the initial point to "
        "SedumiMu on a cone model,\nso step scale is the only thing the caller still controls here."
    )
    # The first solve of the process pays CUDA and cuDSS init.
    solve_capture(build(), settings(1, 0, 0.9))

    hdr = (
        f"{'':>9} {'IR':>8} {'csr':>4} | {'iters':>6} {'ms/iter':>8} {'barrier':>8} "
        f"{'wall':>7} {'mode':>16} | {'objective':>20} {'vs ref':>9}"
    )
    emit()
    emit(hdr)
    emit("-" * len(hdr))
    rows_all = []
    for label, scale, ir, csr in CELLS:
        first, rows = run_cell(ir, csr, scale)
        if not rows:
            emit(
                f"{label:>9} {IR_NAME[ir]:>8} {csr:>4} |  cache build failed every attempt, "
                "no re-solve to time"
            )
            continue
        if not all(r["reuse"] for r in rows):
            emit(f"  !! {label} IR={ir}: cache was not reused on every re-solve")
        modes = [r["mode"] for r in rows]
        mode_s = f"{modes.count('ok')}/{len(modes)} ok"
        if modes.count("soft"):
            mode_s += f" {modes.count('soft')} soft"
        if modes.count("hard"):
            mode_s += f" {modes.count('hard')} hard"
        rec = dict(
            label=label,
            ir=ir,
            csr=csr,
            iters=med(rows, "iters"),
            ms_per_it=med(rows, "ms_per_it"),
            barrier=med(rows, "barrier_ms"),
            wall=med(rows, "wall_ms"),
            mode=mode_s,
            obj=st.median([r["obj"] for r in rows]),
        )
        rows_all.append(rec)
        emit(
            f"{label:>9} {IR_NAME[ir]:>8} {csr:>4} | {rec['iters']:>6.0f} {rec['ms_per_it']:>8.2f} "
            f"{rec['barrier']:>8.1f} {rec['wall']:>7.1f} {rec['mode']:>16} | {rec['obj']:>20.15f}"
            f" {abs(rec['obj'] - a.ref_obj):>9.2e}"
        )

    emit()
    emit("ms/iter is (last - first Elapsed) / (n - 1) over the log's iteration lines, so it")
    emit("excludes setup and the exit test. barrier and wall are totals, medians over the")
    emit("re-solves, and every one of them is a reuse: the cache-building solve is excluded.")
    emit("Objective is the median; a soft breakdown reports a rolled-back iterate. Note the")
    emit("soft rows' barrier totals are 10 ms-quantized, since cuOpt prints the suboptimal")
    emit("line with two decimals of seconds against the optimal line's three.")
    emit(f"vs ref is the gap to {a.ref_obj:.15f}, ClarabelGPU on this step at the same tolerance.")

    for scale in (0.9, 0.99):
        same = [r for r in rows_all if r["label"].endswith(str(scale))]
        on = next((r for r in same if r["ir"] == 1 and not r["csr"]), None)
        off = next((r for r in same if r["ir"] == 0), None)
        if on is None or off is None:
            continue
        emit()
        emit(
            f"step {scale}: IR off costs {off['ms_per_it']:.2f} ms/iter against "
            f"{on['ms_per_it']:.2f} with GMRES, so refinement is "
            f"{on['ms_per_it'] - off['ms_per_it']:.2f} ms/iter "
            f"({100 * (on['ms_per_it'] - off['ms_per_it']) / on['ms_per_it']:.0f}% of an iteration)."
        )

    if a.out:
        with open(a.out, "w") as fh:
            fh.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
