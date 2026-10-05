#!/usr/bin/env python3
"""Reproducibility of repeated cold solves, across initial-point strategy x cuDSS determinism.

One instance (SEQ[1] by default) solved N times cold per cell. The question is whether the runs
agree with each other, so objectives are compared for exact equality rather than to a tolerance.

barrier_dual_initial_point: 0 LustigMarstenShanno, 1 DualLeastSquares, 2 SedumiMu. Automatic (-1)
resolves to SedumiMu on a cone model, so cell 2 is what the default does, pinned for clarity.
Everything else is default except barrier_presolve_bound_free_variables=0.
"""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "cuopt_socp_resolve_cache.py")

ap = argparse.ArgumentParser()
ap.add_argument("--repeat", type=int, default=10)
ap.add_argument("--step", type=int, default=1, help="which mu of the sequence to re-solve")
ap.add_argument("--logs", default=os.path.join(HERE, "logs", "repeat10_strategy_det"))
ap.add_argument(
    "--csr-ir-matvec", type=int, default=0, help="refinement matvec as one SpMV over the KKT CSR"
)
a = ap.parse_args()

STRATEGIES = [(2, "SedumiMu"), (0, "LustigMarstenShanno"), (1, "DualLeastSquares")]
DETS = [0, 1]

ITERS = re.compile(r"iterations: (\d+) distinct value\(s\) \[([^\]]*)\]")
OBJS = re.compile(r"objectives: (\d+) distinct value\(s\), spread (\S+) \((.+?)\)")
OUTCOMES = re.compile(r"outcomes: (\d+) optimal, (\d+) soft, (\d+) hard")
ROW = re.compile(r"^\s*(\d+) \|\s+(\S+)\s+(\S+)\s+(\S+) \|", re.M)


def run(ip, det, name):
    out_dir = os.path.join(a.logs, name)
    cmd = [
        sys.executable, HARNESS,
        "--configs", "default",
        "--repeat", str(a.repeat), "--repeat-step", str(a.step),
        "--initial-point", str(ip),
        "--bound-free-vars", "0",
        "--cudss-deterministic", str(det),
        "--csr-ir-matvec", str(a.csr_ir_matvec),
        "--save-logs", out_dir,
    ]
    p = subprocess.run(
        cmd, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    )
    text = p.stdout + p.stderr
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(" ".join(cmd) + "\n\n" + text)

    it, ob, oc = ITERS.search(text), OBJS.search(text), OUTCOMES.search(text)
    its, mss = [], []
    for row in ROW.finditer(text):
        if row.group(2) != "-":
            its.append(int(row.group(2)))
        if row.group(3) != "-":
            mss.append(float(row.group(3)))
    return dict(
        name=name, ip=ip, det=det,
        n_it=int(it.group(1)) if it else 0,
        it_vals=it.group(2) if it else "",
        n_obj=int(ob.group(1)) if ob else 0,
        spread=ob.group(2) if ob else "",
        identical=(ob.group(3) == "bit-identical") if ob else False,
        optimal=int(oc.group(1)) if oc else 0,
        soft=int(oc.group(2)) if oc else 0,
        hard=int(oc.group(3)) if oc else 0,
        iters=sum(its) / len(its) if its else float("nan"),
        ms=sum(mss) / len(mss) if mss else float("nan"),
        text=text,
    )


rows = []
total = len(STRATEGIES) * len(DETS)
for ip, label in STRATEGIES:
    for det in DETS:
        name = f"ip{ip}-{label}_det{det}"
        print(f"[{len(rows) + 1}/{total}] {name} ...", flush=True)
        r = run(ip, det, name)
        rows.append(r)
        print(
            f"      {r['n_it']} distinct iters [{r['it_vals']}], {r['n_obj']} distinct obj,"
            f" spread {r['spread']}, {'IDENTICAL' if r['identical'] else 'differs'},"
            f" {r['optimal']}/{a.repeat} optimal, {r['ms']:.0f} ms",
            flush=True,
        )

hdr = (
    f"{'initial point':>23} {'cudss_det':>9} | {'reproducible':>12} {'distinct_it':>11}"
    f" {'iterations':>22} {'distinct_obj':>12} {'obj_spread':>10}"
    f" | {'opt':>3} {'soft':>4} {'hard':>4} {'barrier_ms':>10}"
)
lines = [
    "",
    f"{a.repeat} repeated cold solves of the same instance (SEQ[{a.step}]), bound_free_vars=0,",
    "tol=1e-8, IR=1 (GMRES, the default), step scale 0.9 (the default), augmented=1,",
    f"barrier_csr_ir_matvec={a.csr_ir_matvec}.",
    "reproducible = all runs returned the identical iteration count and a bit-identical objective.",
    "",
    hdr,
    "-" * len(hdr),
]
for r in rows:
    name = f"{r['ip']}={dict(STRATEGIES)[r['ip']]}"
    repro = "YES" if r["identical"] and r["n_it"] == 1 else "no"
    lines.append(
        f"{name:>23} {'on' if r['det'] else 'off':>9} |"
        f" {repro:>12} {r['n_it']:>11} {r['it_vals']:>22} {r['n_obj']:>12} {r['spread']:>10} |"
        f" {r['optimal']:>3} {r['soft']:>4} {r['hard']:>4} {r['ms']:>10.0f}"
    )
table = "\n".join(lines)
print(table)
os.makedirs(a.logs, exist_ok=True)
with open(os.path.join(a.logs, "table.txt"), "w") as fh:
    fh.write(table + "\n")
print(f"\nlogs and per-cell summaries under {a.logs}")
