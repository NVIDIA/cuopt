#!/usr/bin/env python3
"""Sweep barrier/cuDSS settings against the warm (cache reuse) barrier time.

The warm path skips convert/presolve/scaling/reordering/symbolic, so its cost is
almost exactly iterations x per-iteration cost. Both are reported, because a
setting that trades one for the other is not an improvement.

Correctness is not assumed: any configuration whose objective drifts from the
baseline by more than the band, or that logs a suboptimal solve, is marked.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(SCRIPT_DIR, "cuopt_socp_resolve_cache.py")

# Applied to every case, so the sweep varies one thing at a time.
BASE = ["--configs", "default", "--initial-point", "2", "--step-scale", "0.99",
        "--cudss-deterministic", "0"]

CASES = [
    ("baseline (ir=fixed_pt)", ["--ir", "2"]),
    # Initial regularization of the augmented KKT system. Automatic (-1) resolves
    # to 1e-8 with cones present for primal, and adaptive_reg ? 1e-8 : 0 for dual.
    ("primal_reg 1e-10", ["--ir", "2", "--set", "barrier_primal_regularization=1e-10"]),
    ("primal_reg 1e-6", ["--ir", "2", "--set", "barrier_primal_regularization=1e-6"]),
    ("primal_reg 1e-4", ["--ir", "2", "--set", "barrier_primal_regularization=1e-4"]),
    ("dual_reg 0", ["--ir", "2", "--set", "barrier_dual_regularization=0"]),
    ("dual_reg 1e-10", ["--ir", "2", "--set", "barrier_dual_regularization=1e-10"]),
    ("dual_reg 1e-6", ["--ir", "2", "--set", "barrier_dual_regularization=1e-6"]),
    # Margin pushing the initial iterate off the cone boundary.
    ("safeguard 0", ["--ir", "2", "--set", "barrier_initial_point_safeguard=0"]),
    ("safeguard 1", ["--ir", "2", "--set", "barrier_initial_point_safeguard=1"]),
    ("safeguard 100", ["--ir", "2", "--set", "barrier_initial_point_safeguard=100"]),
    ("safeguard 1000", ["--ir", "2", "--set", "barrier_initial_point_safeguard=1000"]),
    # The heuristic currently declines Ruiz on this model (norm ratios 1.4); force it.
    ("qcqp_ruiz on", ["--ir", "2", "--set", "qcqp_hyper_ruiz_equilibration=1"]),
    ("qcqp_ruiz off", ["--ir", "2", "--set", "qcqp_hyper_ruiz_equilibration=0"]),
]

ROW = re.compile(r"^default\s+" + r"\s+".join([r"([\d.]+|nan)"] * 4) + r"%?\s*")
SUMMARY = re.compile(
    r"^default\s+([\d.]+|nan)\s+([\d.]+|nan)\s+\+?(-?[\d.]+|nan)\s+([\d.]+|nan)%\s+"
    r"([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)\s+"
    r"([\d.eE+-]+|nan)\s+([\d.eE+-]+|nan)"
)


def num(text):
    return None if text == "nan" else float(text)


def run(name, extra, log_root, keep):
    out_dir = os.path.join(log_root, re.sub(r"[^\w.=+-]+", "_", name))
    cmd = [sys.executable, HARNESS, *BASE, *extra, "--K", str(keep), "--save-logs", out_dir]
    os.makedirs(out_dir, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(text)

    m = next((SUMMARY.match(ln) for ln in text.splitlines() if SUMMARY.match(ln)), None)
    if m is None:
        return dict(name=name, failed=text.strip().splitlines()[-1:] or ["no summary row"])

    cold_wall, warm_wall, _, _, cold_barr, warm_barr, cold_it, warm_it, _, dev, band = (
        num(g) for g in m.groups()
    )
    subopt = sum(
        "Suboptimal solution found" in open(os.path.join(dp, f)).read()
        for dp, _, fs in os.walk(out_dir)
        for f in fs
        if f.endswith(".log")
    )
    return dict(
        name=name,
        warm_barr=warm_barr,
        warm_it=warm_it,
        per_iter=None if not (warm_barr and warm_it) else warm_barr / warm_it,
        cold_barr=cold_barr,
        cold_it=cold_it,
        subopt=subopt,
        dev=dev,
        band=band,
    )


def cell(v, w, prec=1):
    return f"{'-':>{w}}" if v is None else f"{v:>{w}.{prec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--log-root", default=os.path.join(SCRIPT_DIR, "logs", "tune_warm"))
    a = ap.parse_args()

    print(f"portfolio SOCP, {a.K}-step sequence, SedumiMu, cuDSS nondeterministic")
    print("warm_barr is the target metric; per_iter separates cost/iteration from count\n")
    hdr = (f"{'case':24s} {'warm_barr':>10s} {'warm_it':>8s} {'per_iter':>9s} "
           f"{'cold_barr':>10s} {'cold_it':>8s} {'subopt':>7s} {'obj_dev':>9s} {'band':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for name, extra in CASES:
        r = run(name, extra, a.log_root, a.K)
        if "failed" in r:
            print(f"{name:24s}  FAILED: {r['failed'][0][:80]}")
            continue
        flag = "" if r["subopt"] == 0 else "  <-- suboptimal solves"
        print(
            f"{name:24s} {cell(r['warm_barr'], 10)} {cell(r['warm_it'], 8)} "
            f"{cell(r['per_iter'], 9, 2)} {cell(r['cold_barr'], 10)} "
            f"{cell(r['cold_it'], 8)} {r['subopt']:>7d} "
            f"{r['dev']:>9.1e} {r['band']:>9.1e}{flag}"
        )


if __name__ == "__main__":
    main()
