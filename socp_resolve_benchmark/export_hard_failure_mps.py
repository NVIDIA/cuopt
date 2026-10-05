#!/usr/bin/env python3
"""Export one hard-failing mu step of the portfolio SOCP as MPS, and prove the file reproduces it.

At barrier_step_scale=0.99 with iterative refinement off and cuDSS determinism
on, this model fails hard on steps 12, 15 and 18 of the mu sequence, in the
same place every run. Those are usable bug reports only if the exported file
fails the same way, so the export is checked rather than assumed: the script
solves the npz-built model and the reparsed MPS under identical settings and
compares the full per-iteration trace.

Determinism is what makes that comparison meaningful. Without it the two runs
would differ for reasons unrelated to the export.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import re
import tempfile

import numpy as np
import scipy.sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.path.join(SCRIPT_DIR, "data"))
ap.add_argument("--step", type=int, default=12, help="mu step to export; 12, 15 and 18 fail hard")
ap.add_argument("--step-scale", type=float, default=0.99)
ap.add_argument(
    "--initial-point",
    type=int,
    default=1,
    help="barrier_dual_initial_point; 1 (DualLeastSquares) is the other half of the 'tuned' "
    "config these failures were found under, and the step fails only with both set",
)
ap.add_argument("--ir", type=int, default=0)
ap.add_argument("--cudss-deterministic", type=int, default=1)
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--out", default="", help="output path; defaults into --data-dir")
a = ap.parse_args()

import rmm  # noqa: E402

rmm.mr.set_current_device_resource(
    rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), initial_pool_size=1 << 30)
)

from cuopt.linear_programming import (  # noqa: E402
    DataModel,
    ParseMps,
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

OUT = a.out or os.path.join(
    a.data_dir, f"portfolio_socp_n{n}_k{k}_mu_t{a.step:02d}_hardfail.mps"
)


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


def settings():
    s = SolverSettings()
    s.set_parameter(PAR.CUOPT_METHOD, SolverMethod.Barrier)
    s.set_parameter(PAR.CUOPT_AUGMENTED, 1)
    s.set_parameter(PAR.CUOPT_CROSSOVER, False)
    s.set_parameter(PAR.CUOPT_BARRIER_ITERATIVE_REFINEMENT, a.ir)
    s.set_parameter(PAR.CUOPT_BARRIER_PRESOLVE_BOUND_FREE_VARIABLES, 0)
    s.set_parameter(PAR.CUOPT_CUDSS_DETERMINISTIC, bool(a.cudss_deterministic))
    s.set_parameter(PAR.CUOPT_BARRIER_STEP_SCALE, a.step_scale)
    s.set_parameter(PAR.CUOPT_BARRIER_DUAL_INITIAL_POINT, a.initial_point)
    s.set_optimality_tolerance(a.tol)
    return s


_libc = ctypes.CDLL(None)
# Drop the trailing elapsed-seconds column; it is the only nondeterministic field.
_ITER = re.compile(r"^\s*\d+\s+-?\d\.\d+e[+-]\d+\s+-?\d\.\d+e[+-]\d+(?:\s+\S+){3}", re.M)
_HARD = re.compile(r"Search direction computation failed")


def solve_capture(dm, s):
    solve_error = None
    tf = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log")
    old_out, old_err = os.dup(1), os.dup(2)
    os.dup2(tf.fileno(), 1)
    os.dup2(tf.fileno(), 2)
    try:
        # On a non-optimal exit cuOpt leaves the primal solution in expanded cone coordinates
        # (10102 here) instead of user space (5050), and the Python wrapper then refuses to
        # zip it against the model's variable names. Only MPS-parsed models carry names, so
        # this bites the reparsed side only. The log is already complete by then.
        sol = Solve(dm, s)
    except ValueError as exc:
        sol, solve_error = None, exc
    finally:
        _libc.fflush(None)
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        tf.flush()
    log = open(tf.name).read()
    os.unlink(tf.name)
    return sol, log, solve_error


def trace(log):
    lines = _ITER.findall(log)
    return len(lines), hashlib.md5("\n".join(lines).encode()).hexdigest()[:12]


def main():
    s = settings()
    print(
        f"mu step t={a.step}  step_scale={a.step_scale}  initial_point={a.initial_point}  "
        f"IR={a.ir}  cudss_deterministic={a.cudss_deterministic}  tol={a.tol:.0e}"
    )
    solve_capture(build(), s)  # discard: CUDA and cuDSS init

    dm = build()
    _, log_npz, err_npz = solve_capture(dm, s)
    n_npz, h_npz = trace(log_npz)
    print(f"  built from npz:  {n_npz:>3} iterations, hard={bool(_HARD.search(log_npz))}, "
          f"trace {h_npz}")

    dm.writeMPS(OUT)
    size = os.path.getsize(OUT) / 1e6
    print(f"  wrote {OUT} ({size:.1f} MB)")

    _, log_mps, err_mps = solve_capture(ParseMps(OUT), s)
    n_mps, h_mps = trace(log_mps)
    print(f"  reparsed MPS:    {n_mps:>3} iterations, hard={bool(_HARD.search(log_mps))}, "
          f"trace {h_mps}")
    for name, err in (("npz", err_npz), ("MPS", err_mps)):
        if err is not None:
            print(f"    {name} side also raised on solution retrieval: {err}")

    same = h_npz == h_mps
    hard_both = bool(_HARD.search(log_npz)) and bool(_HARD.search(log_mps))
    print(
        f"\n  traces identical: {same}\n"
        f"  hard failure on both: {hard_both}"
    )
    if not hard_both:
        print("  the step did not fail hard this time; try --step 15 or 18")
    return 0 if (same and hard_both) else 1


if __name__ == "__main__":
    raise SystemExit(main())
