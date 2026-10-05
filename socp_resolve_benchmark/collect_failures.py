#!/usr/bin/env python3
"""Collect every cuOpt solve log that hit a numerical breakdown into one file.

A breakdown looks like this in the log: the barrier prints "Numerical error in objective",
then "Restoring previous solution", then "Suboptimal solution found in N iterations".
get_termination_reason() still reports Optimal for these, so the log is the only honest source.

Combined logs (*_all.log, all_runs_combined.log) are skipped so each solve appears once,
and the Clarabel logs are skipped since they are a different solver.
"""
import os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
OUT = os.path.join(ROOT, "numerical_failures_all.log")
SKIP = re.compile(r"(_all\.log$|^all_runs_combined\.log$|^clarabel)")

SUBOPT = re.compile(r"Suboptimal solution found in (\d+) iterations and ([\d.]+) seconds")
OBJ = re.compile(r"^Objective (-?[\d.e+-]+)", re.M)
DUAL = re.compile(r"^Dual infeasibility   \(abs/rel\): \S+/(\S+)", re.M)
ITER = re.compile(r"^\s*(\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", re.M)


def scan(path):
    text = open(path, errors="replace").read()
    if "Numerical error" not in text and "Suboptimal" not in text:
        return None
    m = SUBOPT.search(text)
    o, d = OBJ.search(text), DUAL.search(text)
    rows = ITER.findall(text)
    # The breakdown signature: dual infeasibility crossing back over 1e-8 near convergence
    # while primal and complementarity keep shrinking. Report its peak over the last 6 iters.
    tail = [float(r[4]) for r in rows[-6:]] if rows else []
    return dict(
        path=os.path.relpath(path, ROOT),
        iters=int(m.group(1)) if m else None,
        secs=float(m.group(2)) if m else None,
        obj=o.group(1) if o else "",
        dual_rel=d.group(1) if d else "",
        dual_peak=max(tail) if tail else None,
        # Two distinct endings: the barrier either rolls back to the last good iterate and
        # reports Suboptimal, or the Newton solve itself produces nan and the solve aborts.
        mode="hard" if "Search direction computation failed" in text else "soft",
        restored="Restoring previous solution" in text,
        reuse="reusing cache" in text,
        bounded="Bounded" in text and "free variables in presolve" in text,
        text=text,
    )


def series_of(rel):
    for name in ("cold_repeat", "control_repeat", "cold", "warm", "control"):
        if f"/{name}/" in rel:
            return name
    return "standalone"


hits = []
for dirpath, _, names in os.walk(ROOT):
    for n in sorted(names):
        if not n.endswith(".log") or SKIP.search(n):
            continue
        r = scan(os.path.join(dirpath, n))
        if r:
            hits.append(r)
hits.sort(key=lambda r: r["path"])

by_bench, by_series = {}, {}
for r in hits:
    by_bench[r["path"].split("/")[0]] = by_bench.get(r["path"].split("/")[0], 0) + 1
    s = series_of("/" + r["path"])
    by_series[s] = by_series.get(s, 0) + 1

with open(OUT, "w") as fh:
    fh.write("cuOpt SOCP numerical-breakdown logs, collected by collect_failures.py\n")
    fh.write(f"{len(hits)} failing solves out of the per-solve logs under {ROOT}\n\n")
    fh.write("by benchmark:\n")
    for k, v in sorted(by_bench.items(), key=lambda kv: -kv[1]):
        fh.write(f"  {v:>4}  {k}\n")
    fh.write("\nby series:\n")
    for k, v in sorted(by_series.items(), key=lambda kv: -kv[1]):
        fh.write(f"  {v:>4}  {k}\n")
    hard = sum(1 for r in hits if r["mode"] == "hard")
    fh.write(
        f"\nby failure mode:\n  {len(hits) - hard:>4}  soft (rolled back to the last good"
        f" iterate, reported Suboptimal)\n  {hard:>4}  hard (nan residuals, search direction"
        f" computation failed, solve aborted)\n"
    )

    fh.write(f"\n{'':=<120}\nINDEX\n{'':=<120}\n")
    hdr = (
        f"{'#':>4} {'mode':>4} {'it':>3} {'sec':>6} {'objective':>16} {'dual_rel':>10}"
        f" {'dual_peak':>10} {'reuse':>5} {'bnd':>3}  path\n"
    )
    fh.write(hdr + f"{'':-<120}\n")
    for i, r in enumerate(hits, 1):
        fh.write(
            f"{i:>4} {r['mode']:>4} {r['iters'] or 0:>3} {r['secs'] or 0:>6.2f} {r['obj']:>16} {r['dual_rel']:>10}"
            f" {r['dual_peak'] if r['dual_peak'] is not None else 0:>10.2e}"
            f" {'yes' if r['reuse'] else 'no':>5} {'yes' if r['bounded'] else 'no':>3}  {r['path']}\n"
        )

    for i, r in enumerate(hits, 1):
        fh.write(f"\n\n{'':=<120}\n")
        fh.write(f"=== [{i}/{len(hits)}] {r['path']}\n")
        fh.write(
            f"=== {r['mode']} failure, {r['iters']} iters, {r['secs']}s, series={series_of('/' + r['path'])},"
            f" cache_reuse={r['reuse']}, bounded_free_vars={r['bounded']},"
            f" restored_previous={r['restored']}, peak_dual_inf_last6={r['dual_peak']:.2e}\n"
            if r["dual_peak"] is not None
            else f"=== {r['iters']} iters, series={series_of('/' + r['path'])}\n"
        )
        fh.write(f"{'':=<120}\n")
        fh.write(r["text"].rstrip() + "\n")

print(f"{len(hits)} failing logs -> {OUT}")
for k, v in sorted(by_bench.items(), key=lambda kv: -kv[1]):
    print(f"  {v:>4}  {k}")
print("by series:", dict(sorted(by_series.items(), key=lambda kv: -kv[1])))
