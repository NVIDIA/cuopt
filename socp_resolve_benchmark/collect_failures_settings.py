#!/usr/bin/env python3
"""Collect the numerical-breakdown logs from seven named benchmarks, with the settings each ran under.

Settings come from three places, and each line below is tagged with which one, because they are
not equally trustworthy:
  [harness]  the flags passed to cuopt_socp_resolve_cache.py
  [summary]  the header line the harness itself wrote into the benchmark's summary.txt
  [log]      read back out of the solve log, so it is what the solver actually did
Anything the logs cannot confirm is marked as such rather than stated flatly.
"""
import os, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
OUT = os.path.join(ROOT, "failures_with_settings.log")

BENCHMARKS = [
    "final_bfv0_ir2",
    "pr1732_sedumi099_nondet",
    "cudss_deterministic",
    "pr1856_alg2_nondet",
    "default_strategy_nondeterminism",
    "pr1721_ir0",
    "sedumi099_repeat",
]

# The harness flags each benchmark was launched with. K/tol/IR are cross-checked against
# summary.txt below; bound_free_vars and cudss_deterministic are cross-checked against the log.
HARNESS = {
    "final_bfv0_ir2": dict(
        cmd="--configs default --initial-point 2 --step-scale 0.99 --ir 2 --bound-free-vars 0 "
        "--cudss-deterministic 0 --K 20",
        what="the locked configuration: SedumiMu start, step 0.99, fixed-point refinement",
        code="branch add_socp_support_update_apis with PR #1732 and PR #1721 merged",
    ),
    "pr1732_sedumi099_nondet": dict(
        cmd="--configs default --initial-point 2 --step-scale 0.99 --cudss-deterministic 0 --K 20",
        what="re-run of the cold-vs-warm comparison right after merging PR #1732",
        code="branch add_socp_support_update_apis with PR #1732 merged, before PR #1721",
    ),
    "cudss_deterministic": dict(
        cmd="8 identical cold solves of SEQ[1], cudss_deterministic=true",
        what="does CUDSS_CONFIG_DETERMINISTIC_MODE make the solve reproducible (it does not)",
        code="branch add_socp_support_update_apis, before PR #1732 and PR #1721",
    ),
    "pr1856_alg2_nondet": dict(
        cmd="8 identical cold solves of SEQ[1], default step scale and automatic initial point",
        what="same nondeterminism probe re-run on the PR #1856 ALG2 SpMV port",
        code="branch add_socp_support_update_apis with the PR #1856 cusparse_view ALG2 port",
    ),
    "default_strategy_nondeterminism": dict(
        cmd="8 identical cold solves of SEQ[1], stock settings",
        what="baseline nondeterminism probe: identical cold solves diverge by 1-2 ULP",
        code="branch add_socp_support_update_apis, before PR #1732 and PR #1721",
    ),
    "pr1721_ir0": dict(
        cmd="--configs default --initial-point 2 --step-scale 0.99 --ir 0 "
        "--cudss-deterministic 0 --K 6",
        what="validating barrier_iterative_refinement=Off after merging PR #1721; it breaks the model",
        code="branch add_socp_support_update_apis with PR #1732 and PR #1721 merged",
    ),
    "sedumi099_repeat": dict(
        cmd="--configs default --initial-point 2 --step-scale 0.99 --cudss-deterministic 0 --K 20",
        what="repeat of the SedumiMu step-0.99 run to test whether the failures reproduce",
        code="branch add_socp_support_update_apis, before PR #1732 and PR #1721",
    ),
}

# Settings the logs cannot confirm, and how we know them anyway.
CAVEATS = {
    "cudss_deterministic": "step scale and initial point are the stock defaults (0.9, Automatic), "
    "per the note at the top of all_runs_combined.log",
    "pr1856_alg2_nondet": "step scale and initial point are not recorded for this run. The 26-30 "
    "iteration spread matches the default 0.9 (step 0.99 solves SEQ[1] in 21), and the objective "
    "-9.85078696e-02 matches default_strategy_nondeterminism, so it is the same probe.",
    "default_strategy_nondeterminism": "step scale 0.9 and initial point Automatic (which resolves "
    "to SedumiMu on a cone model), per the note at the top of all_runs_combined.log",
}

SUBOPT = re.compile(r"Suboptimal solution found in (\d+) iterations and ([\d.]+) seconds")
OBJ = re.compile(r"^Objective (-?[\d.e+-]+)", re.M)
DUAL = re.compile(r"^Dual infeasibility   \(abs/rel\): \S+/(\S+)", re.M)
ITER = re.compile(r"^\s*(\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", re.M)
STARTED = re.compile(r"Barrier solver started at ([\d.]+) seconds")


def log_evidence(text):
    """Settings the solver itself printed. Warm solves skip the setup block, so some are absent."""
    ev = []
    if "Handling" in text and "free variables directly in augmented system" in text:
        ev.append("barrier_presolve_bound_free_variables = 0 (log: free variables handled directly)")
    elif "free variables in presolve" in text:
        ev.append("barrier_presolve_bound_free_variables = -1 or 1 (log: bounded in presolve)")
    if "cuDSS solve mode            : deterministic" in text:
        ev.append("cudss_deterministic = true (log: cuDSS solve mode deterministic)")
    elif "cuDSS Version" in text:
        ev.append("cudss_deterministic = false (log prints no deterministic solve mode)")
    if "Linear system               : augmented" in text:
        ev.append("barrier_augmented_system = 1 (log: linear system augmented)")
    if "Adaptive regularization enabled" in text:
        ev.append("barrier_adaptive_regularization = on (log)")
    if "Skipping Ruiz equilibration" in text:
        ev.append("Ruiz equilibration skipped by the solver's own norm-ratio test (log)")
    if "reusing cache" in text:
        ev.append("sequence_solve = 1, cache was reused for this solve (log)")
    elif "Reordering time" in text:
        ev.append("full setup: reordering and symbolic factorization ran, so no cache reuse (log)")
    m = STARTED.search(text)
    if m:
        d = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
        ev.append(
            f"build marker: 'started at {m.group(1)} seconds' -> "
            + ("3-decimal, so the %.3f phase-timing change is in" if d == 3 else "2-decimal, so it predates the %.3f phase-timing change")
        )
    return ev


def scan(path):
    text = open(path, errors="replace").read()
    if "Numerical error" not in text and "Suboptimal" not in text:
        return None
    m, o, d = SUBOPT.search(text), OBJ.search(text), DUAL.search(text)
    rows = ITER.findall(text)
    tail = [float(r[4]) for r in rows[-6:]] if rows else []
    return dict(
        path=os.path.relpath(path, ROOT),
        iters=int(m.group(1)) if m else None,
        secs=float(m.group(2)) if m else None,
        obj=o.group(1) if o else "",
        dual_rel=d.group(1) if d else "",
        dual_peak=max(tail) if tail else None,
        mode="hard" if "Search direction computation failed" in text else "soft",
        evidence=log_evidence(text),
        text=text,
    )


def series_of(rel):
    for name in ("cold_repeat", "control_repeat", "cold", "warm", "control"):
        if f"/{name}/" in rel:
            return name
    return "standalone"


def summary_header(bench):
    for cand in (
        os.path.join(ROOT, bench, "summary.txt"),
        os.path.join(ROOT, bench, "run1", "summary.txt"),
    ):
        if os.path.exists(cand):
            return open(cand).readline().strip(), os.path.relpath(cand, ROOT)
    return None, None


found = {}
for bench in BENCHMARKS:
    hits = []
    for dirpath, _, names in os.walk(os.path.join(ROOT, bench)):
        for n in sorted(names):
            if not n.endswith(".log") or n.endswith("_all.log") or n == "all_runs_combined.log":
                continue
            r = scan(os.path.join(dirpath, n))
            if r:
                hits.append(r)
    found[bench] = sorted(hits, key=lambda r: r["path"])

total = sum(len(v) for v in found.values())

with open(OUT, "w") as fh:
    fh.write("cuOpt SOCP numerical-breakdown logs from seven named benchmarks, with settings\n")
    fh.write("collected by collect_failures_settings.py\n")
    fh.write(f"{total} failing solves across {len(BENCHMARKS)} benchmarks\n\n")
    for b in BENCHMARKS:
        soft = sum(1 for r in found[b] if r["mode"] == "soft")
        n = len(found[b])
        fh.write(f"  {n:>3} failure{'' if n == 1 else 's'} ({soft} soft, {n - soft} hard)  {b}\n")

    fh.write(f"\n{'':=<118}\nSETTINGS PER BENCHMARK\n{'':=<118}\n")
    for b in BENCHMARKS:
        h, rel = summary_header(b)
        k = len(found[b])
        fh.write(f"\n{'':-<118}\n{b}   ({k} failing solve{'' if k == 1 else 's'})\n{'':-<118}\n")
        fh.write(f"  purpose   : {HARNESS[b]['what']}\n")
        fh.write(f"  code      : {HARNESS[b]['code']}\n")
        fh.write(f"  [harness] : {HARNESS[b]['cmd']}\n")
        if h:
            fh.write(f"  [summary] : {h}\n              (from {rel})\n")
        else:
            fh.write("  [summary] : no summary.txt; this predates the harness writing one\n")
        if b in CAVEATS:
            fh.write(f"  unrecorded: {CAVEATS[b]}\n")
        if found[b]:
            fh.write("  [log]     : what the solver printed, from the first failing solve here\n")
            for line in found[b][0]["evidence"]:
                fh.write(f"                - {line}\n")

    fh.write(f"\n\n{'':=<118}\nINDEX OF FAILING SOLVES\n{'':=<118}\n")
    fh.write(
        f"{'#':>4} {'mode':>4} {'it':>3} {'sec':>6} {'objective':>16} {'dual_rel':>10}"
        f" {'dual_peak':>10} {'series':>12}  path\n" + f"{'':-<118}\n"
    )
    n = 0
    for b in BENCHMARKS:
        for r in found[b]:
            n += 1
            fh.write(
                f"{n:>4} {r['mode']:>4} {r['iters'] or 0:>3} {r['secs'] or 0:>6.2f} {r['obj']:>16}"
                f" {r['dual_rel']:>10} {r['dual_peak'] if r['dual_peak'] is not None else 0:>10.2e}"
                f" {series_of('/' + r['path']):>12}  {r['path']}\n"
            )

    n = 0
    for b in BENCHMARKS:
        for r in found[b]:
            n += 1
            fh.write(f"\n\n{'':=<118}\n=== [{n}/{total}] {r['path']}\n")
            fh.write(
                f"=== {r['mode']} failure, {r['iters']} iters, series={series_of('/' + r['path'])},"
                f" peak dual infeasibility over the last 6 iters"
                f" {r['dual_peak']:.2e}\n" if r["dual_peak"] is not None else "===\n"
            )
            fh.write(f"=== [harness] {HARNESS[b]['cmd']}\n")
            for line in r["evidence"]:
                fh.write(f"=== [log] {line}\n")
            fh.write(f"{'':=<118}\n{r['text'].rstrip()}\n")

print(f"{total} failing logs -> {OUT}")
for b in BENCHMARKS:
    print(f"  {len(found[b]):>3}  {b}")
