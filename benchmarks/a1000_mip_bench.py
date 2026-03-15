#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Before/after benchmark for the A1000 MIP kernel changes.

Usage
-----
# Capture baseline (before source change + rebuild):
    python benchmarks/a1000_mip_bench.py --tag baseline --out results/baseline.json

# Capture candidate (after rebuild):
    python benchmarks/a1000_mip_bench.py --tag candidate --out results/candidate.json

# Compare:
    python benchmarks/a1000_mip_bench.py --compare results/baseline.json results/candidate.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# cuOpt import — fail early with a helpful message
# ---------------------------------------------------------------------------
try:
    from cuopt.linear_programming import SolverSettings
    from cuopt.linear_programming.mip import Problem as MIPProblem
except ImportError as e:
    sys.exit(
        f"Cannot import cuopt: {e}\n"
        "Make sure the package is installed and LD_LIBRARY_PATH is set:\n"
        "  export LD_LIBRARY_PATH=/path/to/libcuopt/lib64:$LD_LIBRARY_PATH"
    )

# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------
DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "mip"

# Instances with known-feasible solutions; skip infeasibility tests here
INSTANCES = [
    "sudoku.mps",
    "sample.mps",
    "50v-10-free-bound.mps",
    "neos5-free-bound.mps",
    "bb_optimality.mps",
    "cod105_max.mps",
    "presolve_instance.sh",   # skip — it's a shell script, not an MPS file
]
MPS_INSTANCES = [i for i in INSTANCES if i.endswith(".mps")]

# Per-instance time limit (seconds).  Keep short so the bench finishes fast.
TIME_LIMIT = 60.0
WARMUP_RUNS = 1
TIMED_RUNS = 3


def build_settings(time_limit: float) -> SolverSettings:
    """A1000-tuned runtime settings (Python-level, no rebuild required)."""
    s = SolverSettings()
    s.set_parameter("mip_heuristics_only", True)
    s.set_parameter("mip_cut_passes", 3)
    s.set_parameter("num_cpu_threads", 4)
    s.set_parameter("time_limit", time_limit)
    return s


def solve_instance(mps_path: Path, time_limit: float) -> dict:
    """Return solve time and best objective for one instance."""
    problem = MIPProblem()
    problem.read_mps(str(mps_path))
    settings = build_settings(time_limit)

    t0 = time.perf_counter()
    sol = problem.solve(settings)
    elapsed = time.perf_counter() - t0

    obj = sol.get_objective_value() if sol is not None else math.nan
    status = sol.get_termination_status() if sol is not None else "FAILED"
    return {"time_s": elapsed, "objective": obj, "status": str(status)}


def run_benchmark(tag: str, out_path: Path) -> None:
    results = {"tag": tag, "instances": {}}

    for mps in MPS_INSTANCES:
        path = DATASETS_DIR / mps
        if not path.exists():
            print(f"  [skip] {mps} not found at {path}")
            continue

        print(f"  {mps}")

        # warmup
        for _ in range(WARMUP_RUNS):
            solve_instance(path, TIME_LIMIT)

        # timed runs
        times, objs = [], []
        for _ in range(TIMED_RUNS):
            r = solve_instance(path, TIME_LIMIT)
            times.append(r["time_s"])
            objs.append(r["objective"])
            print(f"    run {len(times)}: {r['time_s']:.3f}s  obj={r['objective']:.6g}  {r['status']}")

        results["instances"][mps] = {
            "mean_time_s": sum(times) / len(times),
            "min_time_s": min(times),
            "best_objective": min(o for o in objs if not math.isnan(o)) if any(not math.isnan(o) for o in objs) else math.nan,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")


def compare(baseline_path: Path, candidate_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())

    print(f"\n{'Instance':<35} {'Base (s)':>10} {'Cand (s)':>10} {'Δ time':>9} {'Δ obj':>12}")
    print("-" * 80)

    regressions = []
    for mps, base_r in baseline["instances"].items():
        if mps not in candidate["instances"]:
            print(f"  {mps}: missing from candidate run")
            continue
        cand_r = candidate["instances"][mps]
        bt = base_r["mean_time_s"]
        ct = cand_r["mean_time_s"]
        pct = (ct - bt) / bt * 100 if bt > 0 else 0.0
        bo = base_r["best_objective"]
        co = cand_r["best_objective"]
        obj_delta = "" if math.isnan(bo) or math.isnan(co) else f"{co - bo:+.4g}"
        marker = ""
        if pct > 5:
            marker = "  !! SPEED REGRESSION"
            regressions.append(mps)
        elif pct < -5:
            marker = "  ** speedup"
        print(f"  {mps:<33} {bt:>10.3f} {ct:>10.3f} {pct:>+8.1f}% {obj_delta:>12}{marker}")

    if regressions:
        print(f"\n[FAIL] Speed regressions detected: {regressions}")
        sys.exit(1)
    else:
        print("\n[PASS] No speed regressions (>5%) detected.")


def main() -> None:
    parser = argparse.ArgumentParser(description="A1000 MIP before/after benchmark")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run benchmark and save results")
    run_p.add_argument("--tag", required=True, help="Label (baseline / candidate)")
    run_p.add_argument("--out", required=True, type=Path, help="Output JSON path")

    cmp_p = sub.add_parser("compare", help="Compare two result JSON files")
    cmp_p.add_argument("baseline", type=Path)
    cmp_p.add_argument("candidate", type=Path)

    # Legacy flat-arg style: --tag / --out / --compare
    parser.add_argument("--tag", help="Label for a run")
    parser.add_argument("--out", type=Path, help="Output JSON path for a run")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"),
                        help="Compare two result JSON files")

    args = parser.parse_args()

    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]))
    elif args.tag and args.out:
        run_benchmark(args.tag, args.out)
    elif args.cmd == "run":
        run_benchmark(args.tag, args.out)
    elif args.cmd == "compare":
        compare(args.baseline, args.candidate)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
