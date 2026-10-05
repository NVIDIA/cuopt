#!/usr/bin/env python3
"""Cold-solve sweep over barrier_step_scale x barrier_iterative_refinement x cudss_deterministic.

Everything else is left at its default except barrier_presolve_bound_free_variables=0, which is
what the reuse path requires and therefore what we measure the cold baseline at. The initial point
is left at Automatic; on a cone model that resolves to SedumiMu anyway, so it matches the runs
that pinned it explicitly.

Cold solves only. A configuration with refinement off breaks down hard, and a hard failure at
t=0 leaves no barrier cache, which aborts the reuse series mid-run; cold-only sidesteps that so
every cell of the sweep produces comparable numbers.
"""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "cuopt_socp_resolve_cache.py")

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=20)
ap.add_argument("--logs", default=os.path.join(HERE, "logs", "defaults_and_tuned_step_ir_cudssdet"))
a = ap.parse_args()

STEP_SCALES = [0.9, 0.99]
IR_MODES = [(0, "off"), (1, "gmres"), (2, "fixed")]
CUDSS_DET = [0, 1]

TALLY = re.compile(r"(\d+)/(\d+) optimal, (\d+) soft, (\d+) hard")
ROW = re.compile(r"^\s*(\d+) \|\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", re.M)


def run(step, ir, det, name):
    out_dir = os.path.join(a.logs, name)
    cmd = [
        sys.executable, HARNESS,
        "--configs", "default", "--cold-only",
        "--K", str(a.K),
        "--step-scale", str(step),
        "--ir", str(ir),
        "--bound-free-vars", "0",
        "--cudss-deterministic", str(det),
        "--save-logs", out_dir,
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    text = p.stdout + p.stderr
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(" ".join(cmd) + "\n\n" + text)

    m = TALLY.search(text)
    its, mss = [], []
    for row in ROW.finditer(text):
        it, ms = row.group(2), row.group(3)
        if it != "-":
            its.append(int(it))
        if ms != "-":
            mss.append(float(ms))
    return dict(
        name=name, step=step, ir=ir, det=det,
        n=int(m.group(2)) if m else 0,
        optimal=int(m.group(1)) if m else 0,
        soft=int(m.group(3)) if m else 0,
        hard=int(m.group(4)) if m else 0,
        iters=sum(its) / len(its) if its else float("nan"),
        it_lo=min(its) if its else 0,
        it_hi=max(its) if its else 0,
        ms=sum(mss) / len(mss) if mss else float("nan"),
        crashed=p.returncode not in (0, 1),
        text=text,
    )


rows = []
total = len(STEP_SCALES) * len(IR_MODES) * len(CUDSS_DET)
for step in STEP_SCALES:
    for ir, ir_name in IR_MODES:
        for det in CUDSS_DET:
            name = f"step{step}_ir{ir}-{ir_name}_det{det}"
            print(f"[{len(rows) + 1}/{total}] {name} ...", flush=True)
            r = run(step, ir, det, name)
            rows.append(r)
            print(
                f"      {r['optimal']}/{r['n']} optimal, {r['soft']} soft, {r['hard']} hard,"
                f" {r['iters']:.1f} iters, {r['ms']:.1f} ms"
                + ("  [HARNESS CRASHED]" if r["crashed"] else ""),
                flush=True,
            )

hdr = (
    f"{'step':>5} {'ir':>11} {'cudss_det':>9} | {'solves':>6} {'optimal':>7} {'soft':>5}"
    f" {'hard':>5} | {'iters':>6} {'it_range':>9} {'barrier_ms':>11}"
)
lines = [
    "",
    f"cold solves only, K={a.K}, bound_free_vars=0, initial point Automatic (SedumiMu on cones),",
    "tol=1e-8, augmented=1, everything else default.  iters and barrier_ms average every solve,",
    "taking the count a soft failure reports; hard failures report neither and are excluded.",
    "",
    hdr,
    "-" * len(hdr),
]
for r in rows:
    ir_label = f"{r['ir']}={dict(IR_MODES)[r['ir']]}"
    det_label = "on" if r["det"] else "off"
    it_range = f"{r['it_lo']}-{r['it_hi']}"
    lines.append(
        f"{r['step']:>5} {ir_label:>11} {det_label:>9} |"
        f" {r['n']:>6} {r['optimal']:>7} {r['soft']:>5} {r['hard']:>5} |"
        f" {r['iters']:>6.1f} {it_range:>9} {r['ms']:>11.1f}"
    )
table = "\n".join(lines)
print(table)
with open(os.path.join(a.logs, "table.txt"), "w") as fh:
    fh.write(table + "\n")
print(f"\nlogs and per-config summaries under {a.logs}")
