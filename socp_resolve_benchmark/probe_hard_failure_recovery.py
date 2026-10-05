#!/usr/bin/env python3
"""What a sequence solve does after a hard failure clears the barrier cache.

A hard failure returns a non-optimal status and solve.cpp clears the whole
cache on that path, so the next update_linear_objective has no transform and
raises. This asks what the caller can still do at that point: whether a solve
runs at all, whether it is cold, and crucially which objective it solves.

The binding writes the DataModel objective only after the cache crush
succeeds, so a raising update leaves the model on the previous step's
objective. A caller that catches the exception and re-solves therefore gets an
answer to the wrong problem unless it re-sets the coefficients itself. That is
the claim under test.

Run with --ir 0, which breaks down on this model within a few steps.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import tempfile

import numpy as np
import scipy.sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.path.join(SCRIPT_DIR, "data"))
ap.add_argument("--K", type=int, default=20)
ap.add_argument("--ir", type=int, default=0)
ap.add_argument("--step-scale", type=float, default=0.99)
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument(
    "--recover",
    action="store_true",
    help="run all K steps, recovering from every lost cache instead of stopping at the first. "
    "Shows which steps run and which of them get to reuse",
)
a = ap.parse_args()

import rmm  # noqa: E402

rmm.mr.set_current_device_resource(
    rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), initial_pool_size=1 << 30)
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


def settings():
    s = SolverSettings()
    s.set_parameter(PAR.CUOPT_METHOD, SolverMethod.Barrier)
    s.set_parameter(PAR.CUOPT_AUGMENTED, 1)
    s.set_parameter(PAR.CUOPT_CROSSOVER, False)
    s.set_parameter(PAR.CUOPT_BARRIER_ITERATIVE_REFINEMENT, a.ir)
    s.set_parameter(PAR.CUOPT_BARRIER_PRESOLVE_BOUND_FREE_VARIABLES, 0)
    s.set_parameter(PAR.CUOPT_SEQUENCE_SOLVE, True)
    s.set_parameter(PAR.CUOPT_BARRIER_STEP_SCALE, a.step_scale)
    s.set_optimality_tolerance(a.tol)
    return s


_libc = ctypes.CDLL(None)
_REUSE = re.compile(r"Barrier: reusing cache \(skip convert/presolve/scaling\)")
_HARD = re.compile(r"Search direction computation failed")
_SUB = re.compile(r"Suboptimal solution found in (\d+) iterations")
_OPT = re.compile(r"Optimal solution found in (\d+) iterations")
_PRESOLVE = re.compile(r"^Presolved problem:", re.M)


def solve_capture(dm, s):
    tf = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log")
    old_out, old_err = os.dup(1), os.dup(2)
    os.dup2(tf.fileno(), 1)
    os.dup2(tf.fileno(), 2)
    try:
        sol = Solve(dm, s)
    finally:
        _libc.fflush(None)
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        tf.flush()
    log = open(tf.name).read()
    os.unlink(tf.name)
    return sol, log


def describe(log):
    if _HARD.search(log):
        return "hard"
    if _SUB.search(log):
        return f"soft ({_SUB.search(log).group(1)} it)"
    if _OPT.search(log):
        return f"ok ({_OPT.search(log).group(1)} it)"
    return "?"


def run_recover(s):
    """Every step gets solved, whether or not the cache survived the one before it.

    When the update is rejected the coefficients have to be written by hand, since
    the binding only assigns them after the cache crush succeeds. Without that the
    solve would silently repeat the previous step's objective.
    """
    print("Every step is solved; a lost cache costs a cold solve, not the step.\n")
    hdr = f"{'t':>4} | {'update':>10} {'path':>6} {'outcome':>14} | {'objective':>18}"
    print(hdr)
    print("-" * len(hdr))
    dm = build(cof(SEQ[0]))
    sol, log = solve_capture(dm, s)
    print(
        f"{0:>4} | {'n/a':>10} {'cold':>6} {describe(log):>14} | "
        f"{sol.get_primal_objective():>18.12f}"
    )
    n_cold, n_warm, n_hard = 1, 0, int(bool(_HARD.search(log)))
    for t in range(1, K):
        try:
            dm.update_linear_objective(cof(SEQ[t]))
            accepted = "accepted"
        except Exception:
            # Cache gone: the update was refused, so set the objective directly.
            dm.set_objective_coefficients(cof(SEQ[t]))
            accepted = "refused"
        sol, log = solve_capture(dm, s)
        warm = bool(_REUSE.search(log))
        n_warm += warm
        n_cold += not warm
        n_hard += bool(_HARD.search(log))
        print(
            f"{t:>4} | {accepted:>10} {'warm' if warm else 'cold':>6} {describe(log):>14} | "
            f"{sol.get_primal_objective():>18.12f}"
        )
    print(f"\n  {K}/{K} steps solved: {n_warm} reused the cache, {n_cold} were cold")
    print(f"  {n_hard} hard failures, each costing the following step its reuse")
    return 0


def main():
    print(f"IR={a.ir} step_scale={a.step_scale} tol={a.tol:.0e}, sequence_solve on throughout")
    s = settings()
    solve_capture(build(cof(SEQ[0])), s)  # discard: CUDA and cuDSS init
    if a.recover:
        return run_recover(s)

    dm = build(cof(SEQ[0]))
    sol, log = solve_capture(dm, s)
    print(f"  t=0  cache build      {describe(log):>14}  reuse={bool(_REUSE.search(log))}")

    failed_at = None
    for t in range(1, K):
        try:
            dm.update_linear_objective(cof(SEQ[t]))
        except Exception as exc:
            print(f"  t={t}  update raised: {type(exc).__name__}")
            print(f"       {exc}")
            failed_at = t
            break
        sol, log = solve_capture(dm, s)
        print(f"  t={t}  reuse solve       {describe(log):>14}  reuse={bool(_REUSE.search(log))}")
    if failed_at is None:
        print("  the series completed; no hard failure to probe. Re-run, it is nondeterministic.")
        return 1

    t = failed_at
    print(f"\n--- state after the update for t={t} raised ---")

    # Q1: did the raising update leave the model objective behind?
    live = np.asarray(dm.get_objective_coefficients(), dtype=np.float64)
    print(
        f"  DataModel objective matches t={t} (requested): "
        f"{np.array_equal(live, cof(SEQ[t]))}"
    )
    print(
        f"  DataModel objective matches t={t - 1} (previous): "
        f"{np.array_equal(live, cof(SEQ[t - 1]))}"
    )

    # Q2: does a solve still run, and is it cold?
    sol2, log2 = solve_capture(dm, s)
    print(
        f"\n  Solve() after the failure: {describe(log2)}, reuse={bool(_REUSE.search(log2))}, "
        f"presolve ran={bool(_PRESOLVE.search(log2))}"
    )
    print(f"  it reported objective {sol2.get_primal_objective():.15f}")

    # Q3: which problem did it actually answer? Compare against cold solves of both.
    s_cold = settings()
    s_cold.set_parameter(PAR.CUOPT_SEQUENCE_SOLVE, False)
    for label, idx in ((f"t={t - 1} (previous)", t - 1), (f"t={t} (requested)", t)):
        ref, rlog = solve_capture(build(cof(SEQ[idx])), s_cold)
        print(
            f"    cold solve of {label:>18}: {ref.get_primal_objective():.15f}"
            f"  gap {abs(ref.get_primal_objective() - sol2.get_primal_objective()):.2e}"
        )

    # Q4: does re-setting the coefficients by hand recover the intended problem?
    dm2 = build(cof(SEQ[t - 1]))
    sol_seed, _ = solve_capture(dm2, s)
    for tt in range(t, K):
        try:
            dm2.update_linear_objective(cof(SEQ[tt]))
        except Exception:
            dm2.set_objective_coefficients(cof(SEQ[tt]))
            fixed, flog = solve_capture(dm2, s)
            ref, _ = solve_capture(build(cof(SEQ[tt])), s_cold)
            print(
                f"\n  recovery by set_objective_coefficients then Solve: {describe(flog)}, "
                f"reuse={bool(_REUSE.search(flog))}"
            )
            print(
                f"    gap to a cold solve of the requested t={tt}: "
                f"{abs(fixed.get_primal_objective() - ref.get_primal_objective()):.2e}"
            )
            break
        solve_capture(dm2, s)

    # Q5: is the cache back, so the sequence can continue?
    try:
        dm.update_linear_objective(cof(SEQ[t]))
        print("\n  update after the recovery solve: accepted, the cache was rebuilt")
        sol3, log3 = solve_capture(dm, s)
        print(f"  next reuse solve: {describe(log3)}, reuse={bool(_REUSE.search(log3))}")
    except Exception as exc:
        print(f"\n  update after the recovery solve still raises: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
