#!/usr/bin/env python3
"""ClarabelGPU SOCP re-solve benchmark (the exact methodology used in our cuOpt comparison).

Problem (risk-budget portfolio, factor model, n=5000 assets, k=50 factors):
    min  -mu'x
    s.t. sum(x) = 1,  F'x - y = 0,  0 <= x <= cap,
         || (sqrt(d) .* x, y) ||_2 <= sigma          (second-order cone, dim 1+n+k = 5051)

Standard conic form  min 1/2 z'Pz + q'z  s.t.  Az + s = b,  s in K,  z = [x; y], P = 0:
    K = Zero(1+k) x Nonneg(2n) x SOC(1+n+k)
    rows: [budget; F'x - y] (Zero) | [-x; x] with b=[0; cap] (Nonneg) | [0; -sqrt(d).*x; -y] with b=[sigma; 0; 0] (SOC)

Timing = the solver's internal solve() timer (host clock around the IPM loop, results synced back).
Sequence: 1 cold solve, then K re-solves where only q = [-mu_t; 0] changes (update_q -> no KKT re-assembly,
symbolic factorization reused; with --ws the previous iterate is also reused = IPM warm start).

Requires the ClarabelGPU Python extension (clarabel_gpu.abi3.so) from the DiffSolver repo on PYTHONPATH.
"""
import argparse, os, sys, time, statistics as st
import numpy as np, scipy.sparse as sp
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ap.add_argument("--K", type=int, default=20, help="number of re-solves (<= 20 available in resolve_seq.npz)")
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--ws", type=int, default=1, help="warm_start_enable (IPM iterate reuse)")
ap.add_argument("--ir-adaptive", type=int, default=1, help="iterative refinement only in the polish window")
ap.add_argument("--equil", type=int, default=1, help="Ruiz equilibration")
ap.add_argument("--check", action="store_true", help="recompute true objective/feasibility from raw data")
ap.add_argument("--verbose", type=int, default=0, help="per-iteration IPM log for every solve")
a = ap.parse_args()
from clarabel_gpu import ClarabelGPU, ZeroConeT, NonnegativeConeT, SecondOrderConeT

D = np.load(f"{a.data_dir}/portfolio.npz"); F, d, mu = D["F"], D["d"], D["mu"]
n, k, cap = int(D["n"]), int(D["k"]), float(D["cap"])
sigma2 = float(np.load(f"{a.data_dir}/portfolio_socp.npz")["sigma2"]); sigma = np.sqrt(sigma2)
SEQ = np.load(f"{a.data_dir}/resolve_seq.npz")["mu_seq"]; K = min(a.K, SEQ.shape[0])

P = sp.csr_matrix((n + k, n + k))
bud = sp.hstack([sp.csr_matrix(np.ones((1, n))), sp.csr_matrix((1, k))])
cou = sp.hstack([sp.csr_matrix(F.T), -sp.eye(k)])
lo = sp.hstack([-sp.eye(n), sp.csr_matrix((n, k))]); up = sp.hstack([sp.eye(n), sp.csr_matrix((n, k))])
soc_head = sp.csr_matrix((1, n + k))
soc_x = sp.hstack([sp.diags(-np.sqrt(d)), sp.csr_matrix((n, k))])
soc_y = sp.hstack([sp.csr_matrix((k, n)), -sp.eye(k)])
A = sp.vstack([bud, cou, lo, up, soc_head, soc_x, soc_y], format="csr")
b = np.concatenate([[1.0], np.zeros(k), np.zeros(n), np.full(n, cap), [sigma], np.zeros(n), np.zeros(k)])
cones = [ZeroConeT(1 + k), NonnegativeConeT(2 * n), SecondOrderConeT(1 + n + k)]
qof = lambda m: np.concatenate([-m, np.zeros(k)])
settings = dict(verbose=bool(a.verbose), max_iter=500, tol_feas=a.tol, tol_gap_abs=a.tol, tol_gap_rel=a.tol,
                equilibrate_enable=bool(a.equil), warm_start_enable=bool(a.ws), ir_adaptive=bool(a.ir_adaptive))

def true_obj(z, m=None): return float(-(mu if m is None else m) @ np.asarray(z)[:n])
def feas(z):
    x = np.asarray(z)[:n]
    return abs(float(x.sum()) - 1.0), max(float(np.max(np.maximum(0, -x))), float(np.max(np.maximum(0, x - cap)))), \
           max(0.0, (float(np.sum(d * x * x) + np.sum((F.T @ x) ** 2)) - sigma2) / sigma2)

print(f"ClarabelGPU SOCP  n={n} k={k} sigma^2={sigma2:.6e}  A: {A.shape} nnz={A.nnz}  cones={[type(c).__name__ for c in cones]}")
print(f"settings: tol={a.tol:.0e} warm_start={bool(a.ws)} ir_adaptive={bool(a.ir_adaptive)} equilibrate={bool(a.equil)}  K={K}")
w = ClarabelGPU(); w.setup(P, qof(mu), A, b, cones, **dict(settings, verbose=False)); w.solve()  # CUDA/cuDSS warm-up, discarded
def banner(i):
    if a.verbose:
        print(f"\n{'='*100}\n=== {i}\n{'='*100}", flush=True)
def row(i, r, m=None):
    extra = ""
    if a.check:
        z = r["x"]; bv, xv, cv = feas(z); extra = f"  {true_obj(z, m):.12f}  {bv:.1e} {xv:.1e} {cv:.1e}"
    print(f"{i:>4} {r['solve_time']*1e3:>9.2f} {r['iterations']:>5} {r['status']:>8} {r['obj_val']:>18.12f}{extra}", flush=True)
hdr = f"\n{'step':>4} {'solve_ms':>9} {'iters':>5} {'status':>8} {'obj':>18}" + ("  true_obj  budget  box  cone" if a.check else "")
if not a.verbose: print(hdr)
s = ClarabelGPU()
banner("cold  (setup + first solve)")
t0 = time.perf_counter(); s.setup(P, qof(mu), A, b, cones, **settings); t_setup = (time.perf_counter() - t0) * 1e3
r = s.solve()
if a.verbose: print(hdr)
row("cold", r); print(f"      (setup {t_setup:.1f} ms incl. cuDSS symbolic analysis)")
res = []
for t in range(K):
    banner(f"re-solve t={t:02d}  (update_q, warm_start={bool(a.ws)})")
    s.update_q(qof(SEQ[t])); r = s.solve(); res.append(r)
    if a.verbose: print(hdr)
    row(t, r, SEQ[t])
ms = [r["solve_time"] * 1e3 for r in res]; its = [r["iterations"] for r in res]
print(f"\nre-solve: mean {st.mean(ms):.2f} ms  min {min(ms):.2f}  max {max(ms):.2f}  | iters mean {st.mean(its):.1f} | all solved: {all(r['status']=='solved' for r in res)}")
