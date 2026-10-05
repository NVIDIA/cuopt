#!/usr/bin/env python3
"""Per-step Clarabel vs cuOpt table for the 20-step mu re-solve sequence.

Both sides re-solve the same instance after only the linear objective changes, and both
reuse setup plus symbolic factorization while restarting the interior-point iterate from
a standard starting point. So the two series measure the same thing.

Clarabel numbers come from its own stdout; cuOpt's come from the per-solve logs the
cache harness saves, which is where its iteration count and barrier timer live.
"""

import argparse
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")

ap = argparse.ArgumentParser()
ap.add_argument("--clarabel", default=os.path.join(LOGS, "clarabel_headtohead.log"))
ap.add_argument("--cuopt-root", default=os.path.join(LOGS, "headtohead"))
ap.add_argument("--series", default="warm", help="cuOpt series to read: warm (reuse) or cold")
a = ap.parse_args()

# Clarabel prints "<step> <solve_ms> <iters> <status> <obj> ..."; the cold row is labelled.
CLA = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+(\d+)\s+(\S+)\s+(-?[\d.]+)")

# cuOpt's barrier log. A clean solve reports both; a soft breakdown reports the Suboptimal
# line instead and a hard one reports neither.
OPT = re.compile(r"Optimal solution found in (\d+) iterations and ([\d.]+)s")
SUB = re.compile(r"Suboptimal solution found in (\d+) iterations and ([\d.]+) seconds")
HARD = re.compile(r"Search direction computation failed")
# The final objective line of the barrier log.
OBJ = re.compile(r"^\s*\d+\s+(-?\d\.\d+e[+-]\d+)\s", re.M)


def read_clarabel(path):
    out = {}
    for line in open(path):
        m = CLA.match(line)
        if m:
            out[int(m.group(1))] = dict(
                ms=float(m.group(2)),
                iters=int(m.group(3)),
                status=m.group(4),
                obj=float(m.group(5)),
            )
    return out


def read_cuopt(path):
    if not os.path.exists(path):
        return None
    log = open(path, errors="replace").read()
    m, sub = OPT.search(log), SUB.search(log)
    objs = OBJ.findall(log)
    return dict(
        iters=int(m.group(1)) if m else (int(sub.group(1)) if sub else None),
        ms=float(m.group(2)) * 1e3 if m else (float(sub.group(2)) * 1e3 if sub else None),
        # Blank for a clean solve so the table only flags the ones that broke down.
        mode="hard" if HARD.search(log) else ("soft" if sub else ""),
        obj=float(objs[-1]) if objs else float("nan"),
    )


def read_series(root, config):
    d = os.path.join(root, config, a.series)
    return {t: read_cuopt(os.path.join(d, f"t{t:02d}.log")) for t in range(20)}


cla = read_clarabel(a.clarabel)
COLS = [
    ("cuOpt default", os.path.join(a.cuopt_root, "csr0"), "default"),
    ("cuOpt tuned", os.path.join(a.cuopt_root, "csr0"), "tuned"),
    ("cuOpt tuned +csrIR", os.path.join(a.cuopt_root, "csr1"), "tuned"),
]
series = [(name, read_series(root, cfg)) for name, root, cfg in COLS]


def fmt(r):
    if r is None or r["iters"] is None:
        return f"{'-':>4} {'-':>8} {'':<4}"
    return f"{r['iters']:>4} {r['ms']:>8.1f} {r['mode']:<4}"


print(f"\nPer-step re-solve, cuOpt series = {a.series}. ms is each solver's own timer:")
print("Clarabel reports total solve time, cuOpt reports barrier time (both exclude setup).\n")
head = f"{'t':>3} | {'Clarabel':>18} |"
for name, _ in series:
    head += f" {name:>22} |"
head += f" {'obj spread':>10}"
print(head)
print("-" * len(head))

rows = []
for t in range(20):
    c = cla.get(t)
    line = f"{t:>3} | {c['iters']:>4} {c['ms']:>8.1f} {'':<4} |" if c else f"{t:>3} | {'-':>18} |"
    objs = [c["obj"]] if c else []
    for _, s in series:
        r = s.get(t)
        line += f" {fmt(r):>22} |"
        if r and r["iters"] is not None:
            objs.append(r["obj"])
    spread = (max(objs) - min(objs)) if len(objs) > 1 else float("nan")
    print(line + f" {spread:>10.2e}")
    rows.append((c, [s.get(t) for _, s in series], spread))


def agg(vals):
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else float("nan")


print("-" * len(head))
cla_ok = sum(1 for c in cla.values() if c["status"] == "solved" and c is not None)
summary = (
    f"{'mean':>3} | {agg([c['iters'] for c, _, _ in rows if c]):>4.1f}"
    f" {agg([c['ms'] for c, _, _ in rows if c]):>8.1f} {'':<4} |"
)
for i, (name, s) in enumerate(series):
    summary += (
        f" {agg([r['iters'] for r in [x[1][i] for x in rows] if r]):>4.1f}"
        f" {agg([r['ms'] for r in [x[1][i] for x in rows] if r]):>8.1f} {'':<4} |"
    )
print(summary)

print("\noutcomes over the 20 re-solves:")
print(f"  {'Clarabel':>22}: {cla_ok}/20 solved")
for i, (name, s) in enumerate(series):
    got = [x[1][i] for x in rows]
    soft = sum(1 for r in got if r and r["mode"] == "soft")
    hard = sum(1 for r in got if r and r["mode"] == "hard")
    print(f"  {name:>22}: {20 - soft - hard}/20 optimal, {soft} soft, {hard} hard")

print("\nobjective agreement (max spread across all solvers, per step):")
sp = [s for _, _, s in rows if s == s]
print(f"  median {st.median(sp):.2e}   worst {max(sp):.2e}")
