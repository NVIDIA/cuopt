#!/usr/bin/env python3
"""cuOpt SOCP re-solve benchmark — the mirror image of clarabel_socp_resolve.py.

Same problem, same data, same 20-step mu sequence. The risk cone is given as a quadratic constraint
(cuOpt has no public SOC API): z'Qz <= sigma^2 with Q = diag(d, 1_k) on z=[x;y]; cuOpt converts it to a
second-order cone internally (barrier log shows 10102 variables / 5103 constraints / 1 SOC).

Each re-solve is a full COLD solve: with quadratic constraints, sequence_solve/update_linear_objective is
disabled in cuOpt (cpp/src/dual_simplex/solve.cpp:423-428) — the script demonstrates this with --try-reuse.
Timing = wall time of Solve() (cuOpt's own log timer is within a few % of this for the barrier path).
Use --mps to load the exported MPS instead of building from npz (same numbers).
"""
import argparse, os, sys, time, statistics as st
import numpy as np, scipy.sparse as sp
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ap.add_argument("--mps", default="", help="load portfolio_socp_n5000_k50.mps via ParseMps instead of building from npz")
ap.add_argument("--K", type=int, default=20); ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--ir", type=int, default=1, help="CUOPT_BARRIER_ITERATIVE_REFINEMENT (1 recommended; 0 is unstable on this SOCP)")
ap.add_argument("--augmented", type=int, default=1); ap.add_argument("--tuned", type=int, default=1, help="dual_initial_point=1, step_scale=0.99")
ap.add_argument("--pool-gib", type=float, default=1.0, help="RMM pool size (0 = none)")
ap.add_argument("--try-reuse", action="store_true", help="show that sequence_solve/update_linear_objective raises with a quadratic constraint")
ap.add_argument("--check", action="store_true")
a = ap.parse_args()
if a.pool_gib > 0:
    import rmm; rmm.mr.set_current_device_resource(rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource(), initial_pool_size=int(a.pool_gib * (1 << 30))))
from cuopt.linear_programming import DataModel, SolverSettings, SolverMethod, Solve, ParseMps
from cuopt.linear_programming.solver import solver_parameters as PAR

D = np.load(f"{a.data_dir}/portfolio.npz"); F, d, mu = D["F"], D["d"], D["mu"]
n, k, cap = int(D["n"]), int(D["k"]), float(D["cap"])
sigma2 = float(np.load(f"{a.data_dir}/portfolio_socp.npz")["sigma2"])
SEQ = np.load(f"{a.data_dir}/resolve_seq.npz")["mu_seq"]; K = min(a.K, SEQ.shape[0])
cof = lambda m: np.concatenate([-m, np.zeros(k)]).astype(np.float64)
def build():
    if a.mps: return ParseMps(a.mps)
    bud = sp.hstack([sp.csr_matrix(np.ones((1, n))), sp.csr_matrix((1, k))]); cou = sp.hstack([sp.csr_matrix(F.T), -sp.eye(k)])
    A = sp.vstack([bud, cou], format="csr"); b = np.concatenate([[1.0], np.zeros(k)]); INF = float("inf")
    dm = DataModel(); dm.set_csr_constraint_matrix(A.data.astype(np.float64), A.indices.astype(np.int32), A.indptr.astype(np.int32))
    dm.set_constraint_lower_bounds(b); dm.set_constraint_upper_bounds(b); dm.set_objective_coefficients(cof(mu))
    dm.set_variable_lower_bounds(np.concatenate([np.zeros(n), np.full(k, -INF)])); dm.set_variable_upper_bounds(np.concatenate([np.full(n, cap), np.full(k, INF)]))
    Qd = np.concatenate([d, np.ones(k)]).astype(np.float64); Qi = np.arange(n + k, dtype=np.int32)
    dm.add_quadratic_constraint("risk", None, None, sigma2, Qd, Qi, Qi, "L"); return dm
def settings(seq=False):
    s = SolverSettings(); s.set_parameter(PAR.CUOPT_METHOD, SolverMethod.Barrier); s.set_parameter(PAR.CUOPT_AUGMENTED, a.augmented)
    try: s.set_parameter(PAR.CUOPT_CROSSOVER, False)
    except Exception: pass
    s.set_parameter(PAR.CUOPT_BARRIER_ITERATIVE_REFINEMENT, a.ir); s.set_parameter(PAR.CUOPT_BARRIER_PRESOLVE_BOUND_FREE_VARIABLES, 0)
    s.set_optimality_tolerance(a.tol); s.set_parameter(PAR.CUOPT_LOG_TO_CONSOLE, False)
    if a.tuned: s.set_parameter(PAR.CUOPT_BARRIER_DUAL_INITIAL_POINT, 1); s.set_parameter(PAR.CUOPT_BARRIER_STEP_SCALE, 0.99)
    if seq: s.sequence_solve = True
    return s
def true_obj(z, m=None): return float(-(mu if m is None else m) @ np.asarray(z)[:n])
def feas(z):
    x = np.asarray(z)[:n]
    return abs(float(x.sum()) - 1.0), max(float(np.max(np.maximum(0, -x))), float(np.max(np.maximum(0, x - cap)))), \
           max(0.0, (float(np.sum(d * x * x) + np.sum((F.T @ x) ** 2)) - sigma2) / sigma2)
print(f"cuOpt SOCP  n={n} k={k} sigma^2={sigma2:.6e}  source={'MPS '+a.mps if a.mps else 'npz'}")
print(f"settings: tol={a.tol:.0e} IR={a.ir} augmented={a.augmented} tuned={a.tuned} pool={a.pool_gib} GiB  K={K}  (every re-solve is a cold solve)")
dm = build(); s = settings(); Solve(dm, s)                                     # warm-up, discarded
print(f"\n{'step':>4} {'wall_ms':>9} {'status':>14} {'obj':>18}" + ("  true_obj  budget  box  cone" if a.check else ""))
def run(i, c, m=None):
    dm.set_objective_coefficients(c); t0 = time.perf_counter(); sol = Solve(dm, s); ms = (time.perf_counter() - t0) * 1e3
    z = np.asarray(sol.get_primal_solution(), dtype=np.float64); ok = z.size == n + k
    extra = ""
    if a.check and ok: bv, xv, cv = feas(z); extra = f"  {true_obj(z, m):.12f}  {bv:.1e} {xv:.1e} {cv:.1e}"
    print(f"{i:>4} {ms:>9.1f} {str(sol.get_termination_reason()):>14} {(float(sol.get_primal_objective()) if ok else float('nan')):>18.12f}{extra}"); return ms, str(sol.get_termination_reason())
run("cold", cof(mu)); res = [run(t, cof(SEQ[t]), SEQ[t]) for t in range(K)]
ms = [m for m, _ in res]; ok = sum(r == "Optimal" for _, r in res)
print(f"\nre-solve (cold): mean {st.mean(ms):.1f} ms  min {min(ms):.1f}  max {max(ms):.1f}  | Optimal {ok}/{K}")
if a.try_reuse:
    print("\n--- sequence_solve / update_linear_objective with a quadratic constraint ---")
    dm2 = build(); s2 = settings(seq=True); Solve(dm2, s2)
    try: dm2.update_linear_objective(cof(SEQ[0])); Solve(dm2, s2); print("reuse path worked (unexpected)")
    except Exception as e: print(f"raises: {type(e).__name__}: {e}")
