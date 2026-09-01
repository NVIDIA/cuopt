#!/usr/bin/env python3
"""Augmented + tuned only: RMM off vs sync pool vs CudaAsyncMemoryResource.

Forces CUOPT_AUGMENTED=1, tuned Barrier, IR=0, bound-free vars=0.
Each mode is a fresh process so the resource is set before any GPU alloc.

  CUDA_VISIBLE_DEVICES=1 python resolve_aug_tuned_rmm.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "..", "results")
INNER = os.path.join(SCRIPT_DIR, "resolve_adat_vs_aug_total.py")
os.makedirs(OUT, exist_ok=True)

POOL_GIB = int(os.environ.get("CUOPT_GIGABYTES_PER_PROC", "1"))

KINDS = (
    ("off", "nopool", "cudaMalloc (no pool)"),
    ("pool", f"rmmpool{POOL_GIB}g", f"RMM pool {POOL_GIB} GiB (cudaMalloc)"),
    ("async", "rmmasync", "RMM CudaAsyncMemoryResource"),
)


def run_mode(kind: str, tag: str) -> dict:
    env = os.environ.copy()
    env["CUOPT_BENCH_PATHS"] = "augmented"
    env["CUOPT_BENCH_CONFIGS"] = "tuned"
    env["BENCH_RMM_KIND"] = kind
    env["CUOPT_GIGABYTES_PER_PROC"] = str(POOL_GIB)
    env["BENCH_LOG_ROOT"] = os.path.join(OUT, f"resolve_aug_tuned_{tag}_logs")
    json_path = os.path.join(OUT, f"resolve_adat_vs_aug_total_{tag}.json")
    stdout_path = os.path.join(OUT, f"resolve_aug_tuned_{tag}.stdout.log")
    with open(stdout_path, "w") as log:
        subprocess.run(
            [sys.executable, INNER],
            cwd=SCRIPT_DIR,
            env=env,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    rows = json.load(open(json_path))
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["path"] == "augmented" and row["config"] == "tuned"
    assert row["barrier_iterative_refinement"] == 0
    assert row["barrier_presolve_bound_free_variables"] == 0
    row["stdout_log"] = stdout_path
    row["json_path"] = json_path
    row["tag"] = tag
    return row


def main() -> None:
    print(
        "augmented / tuned  IR=0  bound_free_vars=0  "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    print("Running off / pool / async in separate processes...\n")

    results = []
    for kind, tag, label in KINDS:
        r = run_mode(kind, tag)
        r["label"] = label
        results.append(r)

    def line(r: dict) -> None:
        print(
            f"{r['label']:42s}  baseline={r['baseline_warm_total_mean_ms']:6.1f} ms  "
            f"reuse={r['session_warm_total_mean_ms']:6.1f} ms  "
            f"save={100.0 * r['savings_warm_total_mean_ms'] / r['baseline_warm_total_mean_ms']:5.1f}%  "
            f"iters={r['session_warm_iters_mean']:.2f}  "
            f"cache_reuse={r['cache_reuse_logs']}/{r['n_warm']}"
        )

    print("=== augmented / tuned (warm mean, t=1..19) ===")
    for r in results:
        line(r)
    off = results[0]
    print()
    for r in results[1:]:
        d_b = r["baseline_warm_total_mean_ms"] - off["baseline_warm_total_mean_ms"]
        d_c = r["session_warm_total_mean_ms"] - off["session_warm_total_mean_ms"]
        print(f"Δ {r['tag']:12s} vs off   baseline={d_b:+6.1f} ms  reuse={d_c:+6.1f} ms")
    for r in results:
        print(f"{r['tag']} stdout: {r['stdout_log']}")


if __name__ == "__main__":
    main()
