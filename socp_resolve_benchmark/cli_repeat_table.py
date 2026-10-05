#!/usr/bin/env python3
"""Reproducibility of repeated cuopt_cli solves of one MPS file, across initial point x determinism.

The CLI counterpart of sweep_repeat_strategy_det.py, which drives the Python
harness and builds the model from npz instead of reading a file. Same cells:
three barrier starting points, cuDSS determinism off and on, and the matrix-free
versus CSR refinement matvec.

Reproducibility is judged on the solver's per-iteration trace with the elapsed
column removed, since that column varies by construction. Iteration counts alone
miss cases where runs differ internally but land on the same count, which is
what GMRES does on this model.

The objective spread is measured on the last iteration line rather than the
reported "Objective" line, which the CLI prints to only 8 digits. On a soft
breakdown the last line is the iterate before the rollback rather than the one
reported, but it is still a deterministic function of the run, so it serves as
a reproducibility probe either way.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import statistics as st
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--mps", default=os.path.join(HERE, "data", "portfolio_socp_n5000_k50_mu_t00.mps"))
ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--ir", type=int, default=1, help="1 = GMRES, the default; the sweep used 1")
ap.add_argument("--step-scale", type=float, default=0.9, help="0.9 is the default")
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--csr", default="0,1", help="comma-separated barrier_csr_ir_matvec values")
ap.add_argument("--out", default="")
a = ap.parse_args()

STRATEGIES = [(2, "SedumiMu"), (0, "LustigMarstenShanno"), (1, "DualLeastSquares")]

_ITER = re.compile(
    r"^\s*\d+\s+(-?\d\.\d+e[+-]\d+)\s+-?\d\.\d+e[+-]\d+(?:\s+\S+){3}", re.M
)
_OPT = re.compile(r"Optimal solution found in (\d+) iterations and ([\d.]+)s")
_SUB = re.compile(r"Suboptimal solution found in (\d+) iterations and ([\d.]+) seconds")
_HARD = re.compile(r"Search direction computation failed")


def run(ip, det, csr):
    cmd = [
        "cuopt_cli", a.mps,
        "--augmented", "1",
        "--barrier-iterative-refinement", str(a.ir),
        "--barrier-dual-initial-point", str(ip),
        "--barrier-step-scale", str(a.step_scale),
        "--barrier-presolve-bound-free-variables", "0",
        "--cudss-deterministic", str(det),
        "--barrier-csr-ir-matvec", str(csr),
    ]
    for which in ("primal", "dual", "gap"):
        cmd += [f"--absolute-{which}-tolerance", str(a.tol),
                f"--relative-{which}-tolerance", str(a.tol)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    log = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    opt, sub = _OPT.search(log), _SUB.search(log)
    objs = _ITER.findall(log)
    # Trace without the elapsed column: findall returns the matched prefix only.
    lines = [m.group(0) for m in _ITER.finditer(log)]
    return dict(
        mode="hard" if _HARD.search(log) else ("soft" if sub else "opt"),
        iters=int(opt.group(1)) if opt else (int(sub.group(1)) if sub else None),
        ms=float(opt.group(2)) * 1e3 if opt else (float(sub.group(2)) * 1e3 if sub else None),
        last_obj=float(objs[-1]) if objs else None,
        trace=hashlib.md5("\n".join(lines).encode()).hexdigest()[:10],
    )


def main():
    out = []

    def emit(line=""):
        print(line, flush=True)
        out.append(line)

    emit(f"cuopt_cli x {a.repeats} on {os.path.basename(a.mps)}")
    emit(
        f"tol={a.tol:.0e}, IR={a.ir}, step scale {a.step_scale}, augmented=1, "
        "bound_free_vars=0."
    )
    hdr = (
        f"{'initial point':>23} {'cudss_det':>9} | {'reproducible':>12} {'traces':>6} "
        f"{'distinct_it':>11} {'iterations':>22} {'obj_spread':>10} | "
        f"{'opt':>3} {'soft':>4} {'hard':>4} {'barrier_ms':>10}"
    )
    for csr in [int(c) for c in a.csr.split(",")]:
        emit()
        emit(
            f"csr_ir_matvec={csr} "
            + ("(matrix-free, the default)" if csr == 0 else "(single SpMV over the augmented CSR)")
        )
        emit(hdr)
        for ip, name in STRATEGIES:
            for det in (0, 1):
                rows = [run(ip, det, csr) for _ in range(a.repeats)]
                traces = {r["trace"] for r in rows}
                its = sorted({r["iters"] for r in rows if r["iters"] is not None})
                objs = [r["last_obj"] for r in rows if r["last_obj"] is not None]
                spread = (max(objs) - min(objs)) if objs else float("nan")
                ms = [r["ms"] for r in rows if r["ms"] is not None]
                modes = [r["mode"] for r in rows]
                emit(
                    f"{str(ip) + '=' + name:>23} {'on' if det else 'off':>9} |"
                    f" {'YES' if len(traces) == 1 else 'no':>12} {len(traces):>6}"
                    f" {len(its):>11} {str(its)[1:-1]:>22} {spread:>10.2e} |"
                    f" {modes.count('opt'):>3} {modes.count('soft'):>4}"
                    f" {modes.count('hard'):>4} {st.mean(ms) if ms else float('nan'):>10.0f}"
                )
    emit()
    emit("reproducible/traces are over the per-iteration trace with the elapsed column removed.")
    emit("obj_spread is max-min of the last iteration line's primal objective, so its floor is")
    emit("about 1e-14 at this magnitude. barrier_ms averages only the solves that reported a time.")
    if a.out:
        with open(a.out, "w") as fh:
            fh.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
