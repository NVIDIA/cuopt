#!/usr/bin/env python3
"""Compare ADAT vs forced-augmented cache reuse on the factor-lifted portfolio.

Same problem / resolve sequence as ``resolve_cuopt_session.py``.
For each linear-system path (ADAT via CUOPT_AUGMENTED=0, augmented via =1)
and config (default, tuned):

- baseline: K solves, cache OFF
- cache:  t=0 store, t=1..K-1 ``update_linear_objective`` + reuse (skip convert/presolve/scaling)

Reports **solver total** means only (no wall). Forces IR and bound-free-vars off
(``barrier_iterative_refinement=0``, ``barrier_presolve_bound_free_variables=0``).

An RMM pool is installed as the current device resource by default, matching
the gRPC worker (``CUOPT_GIGABYTES_PER_PROC``, default 1 GiB). Set
``BENCH_RMM_POOL=0`` for raw cudaMalloc. ``BENCH_RMM_STATS=1`` wraps the pool
in a statistics adaptor (do not use for reported timings).

Expects::

CUDA_VISIBLE_DEVICES=1 python resolve_adat_vs_aug_total.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics as st
import tempfile

import numpy as np
import rmm
import scipy.sparse as sp
from cuopt.linear_programming import DataModel, SolverSettings, SolverMethod, Solve
from cuopt.linear_programming.solver import solver_parameters as PAR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "..", "results")
LOG_ROOT = os.environ.get(
    "BENCH_LOG_ROOT", os.path.join(OUT, "resolve_adat_vs_aug_logs")
)
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)

D = np.load(os.path.join(SCRIPT_DIR, "portfolio.npz"))
F, d, mu = D["F"], D["d"], D["mu"]
n, k, gamma, cap = int(D["n"]), int(D["k"]), float(D["gamma"]), float(D["cap"])
SEQ = np.load(os.path.join(SCRIPT_DIR, "resolve_seq.npz"))["mu_seq"]
K = int(SEQ.shape[0])
TOL = 1e-8
OBJ_ATOL = float(os.environ.get("CUOPT_OBJ_VERIFY_ATOL", "1e-5"))
OBJ_RTOL = float(os.environ.get("CUOPT_OBJ_VERIFY_RTOL", "1e-8"))

Q = sp.diags(np.concatenate([gamma * d, gamma * np.ones(k)]), format="csr")
budget = sp.hstack([sp.csr_matrix(np.ones((1, n))), sp.csr_matrix((1, k))])
couple = sp.hstack([sp.csr_matrix(F.T), -sp.eye(k)])
A = sp.vstack([budget, couple], format="csr")
b = np.concatenate([[1.0], np.zeros(k)])
INF = float("inf")
vlb = np.concatenate([np.zeros(n), np.full(k, -INF)])
vub = np.concatenate([np.full(n, cap), np.full(k, INF)])

_REUSE = re.compile(r"Barrier: reusing cuDSS symbolic analysis \(sparsity hash match\)")
_CACHE_REUSE = re.compile(r"Barrier: reusing cache \(skip convert/presolve/scaling\)")
_STORE_ADAT = re.compile(r"Barrier: stored ADAT symbolic cache hash=0x[0-9a-f]+")
_STORE_AUG = re.compile(r"Barrier: stored augmented symbolic cache hash=0x[0-9a-f]+")
_LINEAR = re.compile(r"Linear system\s+:\s+(\w+)")
_DENSE = re.compile(r"Dense columns\s+:\s+(\d+)")
_OPT = re.compile(r"Optimal solution found in (\d+) iterations and ([\d.]+)s")
_PHASE_PROFILE = re.compile(r"Phase profile: (P\d+ [^0-9\n]+) ([\d.]+)$", re.MULTILINE)
_CACHE_DETAIL = re.compile(r"Cache detail: ([CD]\d+ [^0-9\n]+) ([\d.]+)$", re.MULTILINE)

# CUOPT_AUGMENTED: 0 = ADAT, 1 = augmented
# Optional filters: CUOPT_BENCH_PATHS=augmented CUOPT_BENCH_CONFIGS=tuned
PATHS = (("ADAT", 0), ("augmented", 1))
CONFIGS = (("default", False), ("tuned", True))

# BENCH_RMM_KIND: off | pool | async
#   off   — RMM default (cudaMalloc / cudaFree)
#   pool  — PoolMemoryResource(CudaMemoryResource)  [sync cudaMalloc slab]
#   async — CudaAsyncMemoryResource (cudaMallocAsync / CUDA mempool)
# BENCH_RMM_POOL=0 still means off (compat). Default kind is pool.
_kind_env = os.environ.get("BENCH_RMM_KIND", "").strip().lower()
if _kind_env in ("off", "pool", "async"):
    RMM_KIND = _kind_env
elif os.environ.get("BENCH_RMM_POOL", "1").lower() in ("0", "false", "no"):
    RMM_KIND = "off"
else:
    RMM_KIND = "pool"
RMM_POOL = RMM_KIND != "off"
POOL_GIB = int(os.environ.get("CUOPT_GIGABYTES_PER_PROC", "1"))
RMM_STATS = os.environ.get("BENCH_RMM_STATS", "0").lower() in ("1", "true", "yes")
_MR_KEEPALIVE: list = []


def _rmm_suffix() -> str:
    if RMM_KIND == "off":
        return "_nopool"
    if RMM_KIND == "async":
        return "_rmmasync"
    return f"_rmmpool{POOL_GIB}g"


def _rmm_label() -> str:
    if RMM_KIND == "off":
        return "cudaMalloc (no pool)"
    if RMM_KIND == "async":
        return "RMM CudaAsyncMemoryResource (cudaMallocAsync)"
    return f"RMM pool {POOL_GIB} GiB (cudaMalloc)"


def init_rmm_pool():
    """Install the requested RMM device resource before the first GPU alloc."""
    if RMM_KIND == "off":
        return None
    if RMM_KIND == "async":
        mr = rmm.mr.CudaAsyncMemoryResource()
    else:
        mr = rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(), initial_pool_size=POOL_GIB * (1 << 30)
        )
    _MR_KEEPALIVE.append(mr)
    stats = None
    if RMM_STATS:
        stats = rmm.mr.StatisticsResourceAdaptor(mr)
        _MR_KEEPALIVE.append(stats)
        mr = stats
    rmm.mr.set_current_device_resource(mr)
    return stats


def cof(muv):
    return np.concatenate([-muv, np.zeros(k)]).astype(np.float64)


def build_dm(c):
    dm = DataModel()
    dm.set_csr_constraint_matrix(
        A.data.astype(np.float64), A.indices.astype(np.int32), A.indptr.astype(np.int32)
    )
    dm.set_constraint_lower_bounds(b.astype(np.float64))
    dm.set_constraint_upper_bounds(b.astype(np.float64))
    dm.set_objective_coefficients(c)
    dm.set_quadratic_objective_matrix(
        Q.data.astype(np.float64), Q.indices.astype(np.int32), Q.indptr.astype(np.int32)
    )
    dm.set_variable_lower_bounds(vlb)
    dm.set_variable_upper_bounds(vub)
    return dm


def make_settings(tuned: bool, *, sequence_solve: bool, augmented: int) -> SolverSettings:
    s = SolverSettings()
    s.set_parameter(PAR.CUOPT_METHOD, SolverMethod.Barrier)
    s.set_parameter(PAR.CUOPT_AUGMENTED, augmented)
    try:
        s.set_parameter(PAR.CUOPT_CROSSOVER, False)
    except Exception:
        pass
    s.set_parameter(PAR.CUOPT_BARRIER_ITERATIVE_REFINEMENT, 0)
    s.set_parameter(PAR.CUOPT_BARRIER_PRESOLVE_BOUND_FREE_VARIABLES, 0)
    s.sequence_solve = sequence_solve
    s.set_optimality_tolerance(TOL)
    if tuned:
        s.set_parameter(PAR.CUOPT_BARRIER_DUAL_INITIAL_POINT, 1)
        s.set_parameter(PAR.CUOPT_BARRIER_STEP_SCALE, 0.99)
    return s


def solve_capture(dm, s, log_path: str | None = None):
    tf = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log")
    old_out = os.dup(1)
    old_err = os.dup(2)
    os.dup2(tf.fileno(), 1)
    os.dup2(tf.fileno(), 2)
    try:
        sol = Solve(dm, s)
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        tf.flush()
    log = open(tf.name).read()
    os.unlink(tf.name)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as fh:
            fh.write(log)
    return sol, log


def parse_solve(sol, log: str) -> dict:
    m = _OPT.search(log)
    phase_profile = {label.strip(): float(ms) for label, ms in _PHASE_PROFILE.findall(log)}
    cache_detail = {label.strip(): float(ms) for label, ms in _CACHE_DETAIL.findall(log)}
    return dict(
        total_ms=float(m.group(2)) * 1e3 if m else float("nan"),
        iters=int(m.group(1)) if m else -1,
        status=sol.get_termination_reason(),
        obj=float(sol.get_primal_objective()),
        reuse=len(_REUSE.findall(log)),
        cache_reuse=len(_CACHE_REUSE.findall(log)),
        store_adat=len(_STORE_ADAT.findall(log)),
        store_aug=len(_STORE_AUG.findall(log)),
        linear_system=(_LINEAR.search(log).group(1) if _LINEAR.search(log) else "unknown"),
        n_dense_columns=int(_DENSE.search(log).group(1)) if _DENSE.search(log) else 0,
        phase_profile_ms=phase_profile,
        cache_detail_ms=cache_detail,
    )


def objs_close(a: float, b: float) -> bool:
    tol = max(OBJ_ATOL, OBJ_RTOL * max(abs(a), abs(b), 1.0))
    return abs(a - b) <= tol


def mean_total(rows: list[dict]) -> float:
    return st.mean(r["total_ms"] for r in rows)


def mean_profile(rows: list[dict], key: str) -> dict[str, float]:
    labels = sorted({label for row in rows for label in row[key]})
    return {
        label: st.mean(row[key].get(label, 0.0) for row in rows)
        for label in labels
    }


def _phase_dir(path_name: str, config: str, phase: str) -> str:
    return os.path.join(LOG_ROOT, f"{path_name}_{config}_{phase}")


def run_baseline(tuned: bool, augmented: int, path_name: str, config: str) -> list[dict]:
    log_dir = _phase_dir(path_name, config, "baseline")
    s = make_settings(tuned, sequence_solve=False, augmented=augmented)
    solve_capture(
        build_dm(cof(mu)),
        s,
        os.path.join(log_dir, "warmup.log"),
    )
    rows = []
    for t in range(K):
        sol, log = solve_capture(
            build_dm(cof(SEQ[t])),
            make_settings(tuned, sequence_solve=False, augmented=augmented),
            os.path.join(log_dir, f"t{t:02d}.log"),
        )
        rows.append(parse_solve(sol, log))
    return rows


def run_session(tuned: bool, augmented: int, path_name: str, config: str) -> list[dict]:
    log_dir = _phase_dir(path_name, config, "cache")
    solve_capture(
        build_dm(cof(mu)),
        make_settings(tuned, sequence_solve=False, augmented=augmented),
        os.path.join(log_dir, "warmup.log"),
    )

    dm = build_dm(cof(SEQ[0]))
    s = make_settings(tuned, sequence_solve=True, augmented=augmented)
    rows = []
    sol, log = solve_capture(dm, s, os.path.join(log_dir, "t00.log"))
    rows.append(parse_solve(sol, log))

    for t in range(1, K):
        dm.update_linear_objective(cof(SEQ[t]))
        sol, log = solve_capture(dm, s, os.path.join(log_dir, f"t{t:02d}.log"))
        rows.append(parse_solve(sol, log))
    return rows


def summarize(path_name: str, config: str, baseline: list[dict], session_rows: list[dict]) -> dict:
    warm_b = baseline[1:]
    warm_s = session_rows[1:]
    b_warm = mean_total(warm_b)
    s_warm = mean_total(warm_s)
    s0 = session_rows[0]
    reuse_warm = sum(r["reuse"] for r in warm_s)
    cache_reuse_warm = sum(r["cache_reuse"] for r in warm_s)
    obj_ok = all(objs_close(baseline[t]["obj"], session_rows[t]["obj"]) for t in range(K))
    max_obj_delta = max(abs(baseline[t]["obj"] - session_rows[t]["obj"]) for t in range(K))
    save = b_warm - s_warm
    iter_warm = st.mean(r["iters"] for r in warm_s)

    got_path = s0["linear_system"]
    assert got_path.lower() == path_name.lower(), (
        f"expected linear_system={path_name}, got {got_path}"
    )
    assert obj_ok, f"{path_name}/{config}: cache objectives differ from baseline"
    assert cache_reuse_warm == len(warm_s), (
        f"{path_name}/{config}: expected update_linear_objective cache reuse on every warm solve, "
        f"got {cache_reuse_warm}/{len(warm_s)}"
    )

    print(
        f"  {path_name:10s} / {config:7s}  "
        f"baseline_warm_total={b_warm:6.1f}  "
        f"session_warm_total={s_warm:6.1f}  "
        f"save={save:+6.1f} ({100.0 * save / b_warm:4.1f}%)  "
        f"cache_reuse={cache_reuse_warm}/{len(warm_s)}  "
        f"hash_reuse={reuse_warm}/{len(warm_s)}  obj_match=YES"
    )
    return dict(
        path=path_name,
        config=config,
        K=K,
        linear_system=got_path,
        n_dense_columns=s0["n_dense_columns"],
        store_adat=s0["store_adat"],
        store_aug=s0["store_aug"],
        baseline_warm_total_mean_ms=b_warm,
        session_warm_total_mean_ms=s_warm,
        savings_warm_total_mean_ms=save,
        reuse_logs=reuse_warm,
        cache_reuse_logs=cache_reuse_warm,
        n_warm=len(warm_s),
        session_warm_iters_mean=iter_warm,
        baseline_cold_phase_profile_ms=baseline[0]["phase_profile_ms"],
        session_cold_phase_profile_ms=session_rows[0]["phase_profile_ms"],
        baseline_warm_phase_profile_mean_ms=mean_profile(warm_b, "phase_profile_ms"),
        session_warm_phase_profile_mean_ms=mean_profile(warm_s, "phase_profile_ms"),
        baseline_cold_cache_detail_ms=baseline[0]["cache_detail_ms"],
        session_cold_cache_detail_ms=session_rows[0]["cache_detail_ms"],
        baseline_warm_cache_detail_mean_ms=mean_profile(warm_b, "cache_detail_ms"),
        session_warm_cache_detail_mean_ms=mean_profile(warm_s, "cache_detail_ms"),
        obj_max_abs_delta=max_obj_delta,
        obj_match=obj_ok,
        barrier_iterative_refinement=0,
        barrier_presolve_bound_free_variables=0,
        rmm_pool=RMM_POOL,
        rmm_kind=RMM_KIND,
        pool_gib=POOL_GIB if RMM_KIND == "pool" else 0,
    )


def print_table(rows: list[dict]) -> None:
    print("\n=== ADAT vs forced augmented (solver total, warm mean, ms) ===")
    hdr = (
        f"{'path':10s} {'config':7s}  "
        f"{'baseline':>10s} {'cache':>10s} {'save':>10s} {'save%':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        b = r["baseline_warm_total_mean_ms"]
        s = r["session_warm_total_mean_ms"]
        sav = r["savings_warm_total_mean_ms"]
        print(
            f"{r['path']:10s} {r['config']:7s}  "
            f"{b:10.1f} {s:10.1f} {sav:+10.1f} {100.0 * sav / b:6.1f}%"
        )


def _selected_paths():
    wanted = os.environ.get("CUOPT_BENCH_PATHS")
    if not wanted:
        return PATHS
    names = {x.strip() for x in wanted.split(",") if x.strip()}
    return tuple(p for p in PATHS if p[0] in names)


def _selected_configs():
    wanted = os.environ.get("CUOPT_BENCH_CONFIGS")
    if not wanted:
        return CONFIGS
    names = {x.strip() for x in wanted.split(",") if x.strip()}
    return tuple(c for c in CONFIGS if c[0] in names)


def _write_combined_solver_log(path_name: str, config: str) -> str:
    out_path = os.path.join(OUT, f"{path_name}_{config}_all_solver.log")
    with open(out_path, "w") as out:
        for phase in ("baseline", "cache"):
            log_dir = _phase_dir(path_name, config, phase)
            out.write(f"\n========== {path_name} / {config} / {phase} ==========\n")
            for name in ["warmup.log"] + [f"t{t:02d}.log" for t in range(K)]:
                p = os.path.join(log_dir, name)
                if not os.path.exists(p):
                    continue
                out.write(f"\n----- {phase}/{name} -----\n")
                out.write(open(p).read())
    return out_path


def main():
    if os.path.isdir(LOG_ROOT):
        shutil.rmtree(LOG_ROOT)
    os.makedirs(LOG_ROOT, exist_ok=True)

    stats = init_rmm_pool()
    alloc = _rmm_label()
    if stats is not None:
        alloc += " + statistics adaptor"

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"portfolio resolve: n={n} k={k} K={K}  CUDA_VISIBLE_DEVICES={cvd}  allocator={alloc}")
    print(
        "Comparing ADAT (CUOPT_AUGMENTED=0) vs forced augmented (=1); totals only. "
        "IR=0 bound_free_vars=0. Session warm solves use update_linear_objective (cache reuse).\n"
    )

    rows = []
    for path_name, aug in _selected_paths():
        for config, tuned in _selected_configs():
            print(f"--- {path_name} / {config}: baseline ---")
            baseline = run_baseline(tuned, aug, path_name, config)
            print(f"--- {path_name} / {config}: cache ---")
            session_rows = run_session(tuned, aug, path_name, config)
            rows.append(summarize(path_name, config, baseline, session_rows))
            combined = _write_combined_solver_log(path_name, config)
            print(f"combined solver log: {combined}")

    print_table(rows)
    if stats is not None:
        counts = stats.allocation_counts
        print(
            f"\nRMM pool traffic: {counts.total_count} allocations, "
            f"peak {counts.peak_bytes / (1 << 20):.1f} MiB, "
            f"current {counts.current_bytes / (1 << 20):.1f} MiB"
        )
    suffix = _rmm_suffix()
    out_path = os.path.join(OUT, f"resolve_adat_vs_aug_total{suffix}.json")
    json.dump(rows, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")
    print(f"solver logs: {LOG_ROOT}")


if __name__ == "__main__":
    main()
