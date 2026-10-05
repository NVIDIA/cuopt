#!/usr/bin/env python3
"""Split every cuOpt numerical breakdown into a soft file and a hard file, settings-annotated.

soft: the barrier hit "Numerical error in objective", restored the last good iterate and
      reported "Suboptimal solution found". get_termination_reason() still says Optimal.
hard: the Newton solve produced nan residuals and "Search direction computation failed".
      No iteration count, nan objective, no barrier cache, get_termination_reason() says
      NumericalError.

Settings above each log are tagged by where they came from:
  [command] the verbatim invocation, when the run recorded one
  [summary] the header line the harness wrote into summary.txt
  [family]  the flags shared by every cell of that sweep
  [knob]    the one thing this cell changed, taken from its directory name
  [log]     read back out of the solve log, so it is what the solver actually did
"""
import argparse, os, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

ap = argparse.ArgumentParser()
ap.add_argument(
    "--scope",
    default="defaults_and_tuned_step_ir_cudssdet",
    help="benchmark directory under logs/ to collect from; empty string walks all of logs/",
)
ap.add_argument("--out-prefix", default="", help="output name prefix, defaults to the scope")
args = ap.parse_args()

WALK = os.path.join(ROOT, args.scope) if args.scope else ROOT
PREFIX = args.out_prefix or (args.scope or "all")
SOFT_OUT = os.path.join(ROOT, f"{PREFIX}_soft_failures.log")
HARD_OUT = os.path.join(ROOT, f"{PREFIX}_hard_failures.log")
SKIP = re.compile(r"(_all\.log$|^all_runs_combined\.log$|^clarabel|_soft_failures|_hard_failures|^numerical_failures|^failures_with_settings)")

# Flags shared by every cell of a sweep, keyed by the top-level log directory.
FAMILY = {
    "defaults_and_tuned_step_ir_cudssdet":
        "cold solves only; step scale, refinement mode and cuDSS determinism are the swept axes; "
        "bound_free_vars=0, initial point Automatic (SedumiMu on cones), everything else default",
    "tune_warm":
        "--initial-point 2 --step-scale 0.99 --cudss-deterministic 0 --bound-free-vars 0 --K 8, "
        "one knob changed per cell",
    "tune_reg":
        "--initial-point 2 --step-scale 0.99 --ir 2 --cudss-deterministic 0 --bound-free-vars 0 "
        "--K 8, one regularization knob changed per cell",
    "ndlevels_ir2":
        "--initial-point 2 --step-scale 0.99 --ir 2 --cudss-deterministic 0 --bound-free-vars 0, "
        "sweeping cudss_hyper_nd_nlevels",
    "bfv_probe":
        "--initial-point 2 --step-scale 0.99 --cudss-deterministic 0, sweeping "
        "barrier_presolve_bound_free_variables",
    "bfv_default_-1":
        "--initial-point 2 --step-scale 0.99 --cudss-deterministic 0 --bound-free-vars -1, the "
        "automatic setting, which bounds free variables in presolve",
    "irtol_1e-10":
        "--initial-point 2 --step-scale 0.99 --cudss-deterministic 0 --bound-free-vars 0 with a "
        "TEMPORARY source edit relaxing the hardcoded sparse-cone refinement tolerance in "
        "barrier.cu from 1e-12 to 1e-10. The edit was reverted; this whole directory is a "
        "deliberately broken configuration",
    "pr1721_ir0": "--initial-point 2 --step-scale 0.99 --ir 0 --cudss-deterministic 0 "
                  "--bound-free-vars 0 --K 6, validating refinement Off after merging PR #1721",
    "pr1721_ir1": "--initial-point 2 --step-scale 0.99 --ir 1 --cudss-deterministic 0 "
                  "--bound-free-vars 0 --K 6",
    "pr1721_ir2": "--initial-point 2 --step-scale 0.99 --ir 2 --cudss-deterministic 0 "
                  "--bound-free-vars 0 --K 6",
    "final_bfv0_ir2": "--initial-point 2 --step-scale 0.99 --ir 2 --bound-free-vars 0 "
                      "--cudss-deterministic 0 --K 20, the locked configuration",
    "pr1732_sedumi099_nondet": "--initial-point 2 --step-scale 0.99 --cudss-deterministic 0 --K 20, "
                               "re-run right after merging PR #1732",
    "sedumi099_repeat": "--initial-point 2 --step-scale 0.99 --cudss-deterministic 0 --K 20, "
                        "repeat to test whether the failures reproduce",
    "sedumi_step099_nondet": "--initial-point 2 --step-scale 0.99 --cudss-deterministic 0 --K 20",
    "dls_step099_nondet": "--initial-point 1 (DualLeastSquares) --step-scale 0.99 "
                          "--cudss-deterministic 0 --K 20; reuse overrides this with SedumiMu, so "
                          "the run also carries a pinned-SedumiMu control series",
    "cudss_deterministic": "8 identical cold solves of SEQ[1] with cudss_deterministic=true; step "
                           "scale 0.9 and initial point Automatic, per all_runs_combined.log",
    "default_strategy_nondeterminism": "8 identical cold solves of SEQ[1] at stock settings: step "
                                       "scale 0.9, initial point Automatic, IR=1, sequence_solve off",
    "pr1856_alg2_nondet": "8 identical cold solves of SEQ[1] on the PR #1856 ALG2 SpMV port. Step "
                          "scale and initial point unrecorded; the 26-30 iteration spread matches "
                          "the 0.9 default and the objective matches default_strategy_nondeterminism",
    "pr1856_alg2_cudss_det": "the PR #1856 ALG2 port with cudss_deterministic=true",
}

# These files get shared, so strip what identifies this host and this checkout. The hardware
# model, CUDA version and cuOpt git hash stay: timings are unreadable without them.
_SCRUB = [
    # Checkout location and conda prefix, in that order so the repo-relative form wins.
    (re.compile(r"\S*/socp_resolve_benchmark/"), "socp_resolve_benchmark/"),
    (re.compile(r"\S*/bin/python\b"), "python"),
    (re.compile(r"/(?:home/nfs|home|raid)/[^/\s]+"), "~"),
    # Identifies the physical GPU.
    (re.compile(r"^CUDA device UUID: .*$\n?", re.M), ""),
    # Host memory pressure when the run happened, not a property of the solve.
    (re.compile(r", RAM usage: [\d.]+/[\d.]+GiB"), ""),
]


def scrub(text):
    for pat, repl in _SCRUB:
        text = pat.sub(repl, text)
    return text


SUBOPT = re.compile(r"Suboptimal solution found in (\d+) iterations and ([\d.]+) seconds")
OBJ = re.compile(r"^Objective (-?[\d.e+-]+)", re.M)
ITER = re.compile(r"^\s*(\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", re.M)


def log_evidence(text):
    ev = []
    if "free variables directly in augmented system" in text:
        ev.append("barrier_presolve_bound_free_variables = 0 (free variables handled directly)")
    elif "free variables in presolve" in text:
        ev.append("barrier_presolve_bound_free_variables = -1 or 1 (bounded in presolve)")
    if "cuDSS solve mode            : deterministic" in text:
        ev.append("cudss_deterministic = true")
    elif "cuDSS Version" in text:
        ev.append("cudss_deterministic = false (no deterministic solve mode printed)")
    if "Linear system               : augmented" in text:
        ev.append("augmented system")
    elif "Linear system" in text:
        ev.append("normal equations / ADAT")
    if "Adaptive regularization enabled" in text:
        ev.append("barrier_adaptive_regularization = on")
    if "reusing cache" in text:
        ev.append("sequence_solve = 1 and the cache was reused for this solve")
    elif "Reordering time" in text:
        ev.append("full setup ran (reordering + symbolic factorization), so no cache reuse")
    return ev


def config_dir(rel):
    """Nearest ancestor holding a summary.txt, else the top-level benchmark directory."""
    parts = rel.split("/")
    for i in range(len(parts) - 1, 0, -1):
        cand = "/".join(parts[:i])
        if os.path.exists(os.path.join(ROOT, cand, "summary.txt")):
            return cand
    return parts[0]


def summary_of(cfg):
    """Returns (verbatim_command, header_line). The sweep driver records the whole command."""
    path = os.path.join(ROOT, cfg, "summary.txt")
    if not os.path.exists(path):
        return None, None
    first = open(path).readline().strip()
    if "cuopt_socp_resolve_cache.py" in first:
        return first, None
    return None, first or None


def series_of(rel):
    for name in ("cold_repeat", "control_repeat", "cold", "warm", "control"):
        if f"/{name}/" in rel:
            return name
    return "standalone"


def scan(path):
    text = open(path, errors="replace").read()
    if "Numerical error" not in text and "Suboptimal" not in text:
        return None
    hard = "Search direction computation failed" in text
    m, o = SUBOPT.search(text), OBJ.search(text)
    rows = ITER.findall(text)
    tail = [float(r[4]) for r in rows[-6:]] if rows else []
    rel = os.path.relpath(path, ROOT)
    return dict(
        path=rel,
        mode="hard" if hard else "soft",
        iters=int(m.group(1)) if m else None,
        secs=float(m.group(2)) if m else None,
        obj=o.group(1) if o else "",
        dual_peak=max(tail) if tail else None,
        last_iter=int(rows[-1][0]) if rows else None,
        cfg=config_dir(rel),
        evidence=log_evidence(text),
        text=text,
    )


hits = []
for dirpath, _, names in os.walk(WALK):
    for n in sorted(names):
        if n.endswith(".log") and not SKIP.search(n):
            r = scan(os.path.join(dirpath, n))
            if r:
                hits.append(r)
hits.sort(key=lambda r: r["path"])


def write(out, mode, rows):
    with open(out, "w") as fh:
        fh.write(f"cuOpt SOCP {mode.upper()} numerical breakdowns, collected by collect_by_mode.py\n")
        fh.write(__doc__.split("Settings above")[0].split("\n", 2)[2].rstrip() + "\n\n")
        fh.write(f"{len(rows)} {mode} failures\n\nby benchmark:\n")
        tally = {}
        for r in rows:
            tally[r["cfg"]] = tally.get(r["cfg"], 0) + 1
        for key, v in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])):
            fh.write(f"  {v:>4}  {key}\n")

        fh.write(f"\n{'':=<118}\nINDEX\n{'':=<118}\n")
        if mode == "soft":
            head = f"{'#':>4} {'it':>3} {'sec':>6} {'objective':>16} {'dual_peak':>10} {'series':>12}  path\n"
        else:
            head = f"{'#':>4} {'last_it':>7} {'dual_peak':>10} {'series':>12}  path\n"
        fh.write(head + f"{'':-<118}\n")
        for i, r in enumerate(rows, 1):
            peak = r["dual_peak"] if r["dual_peak"] is not None else 0
            if mode == "soft":
                fh.write(
                    f"{i:>4} {r['iters'] or 0:>3} {r['secs'] or 0:>6.2f} {r['obj']:>16}"
                    f" {peak:>10.2e} {series_of('/' + r['path']):>12}  {r['path']}\n"
                )
            else:
                fh.write(
                    f"{i:>4} {r['last_iter'] or 0:>7} {peak:>10.2e}"
                    f" {series_of('/' + r['path']):>12}  {r['path']}\n"
                )

        for i, r in enumerate(rows, 1):
            cmd, hdr = summary_of(r["cfg"])
            top, leaf = r["cfg"].split("/")[0], r["cfg"].split("/")[-1]
            fh.write(f"\n\n{'':=<118}\n=== [{i}/{len(rows)}] {r['path']}\n{'':-<118}\n")
            if mode == "soft":
                fh.write(
                    f"=== {r['iters']} iterations, {r['secs']}s, restored objective {r['obj']},"
                    f" series {series_of('/' + r['path'])}\n"
                )
            else:
                fh.write(
                    f"=== aborted after iteration {r['last_iter']}, nan residuals, no objective,"
                    f" series {series_of('/' + r['path'])}\n"
                )
            if r["dual_peak"] is not None:
                fh.write(f"=== peak dual infeasibility over the last 6 iterations {r['dual_peak']:.2e}\n")
            fh.write("=== SETTINGS\n")
            fh.write(f"===   config     {r['cfg']}\n")
            if cmd:
                fh.write(f"===   [command]  {scrub(cmd)}\n")
            if hdr:
                fh.write(f"===   [summary]  {scrub(hdr)}\n")
            if top in FAMILY:
                fh.write(f"===   [family]   {FAMILY[top]}\n")
            # Redundant when the verbatim command is present, since that already names the knob.
            if leaf != top and not cmd:
                fh.write(f"===   [knob]     {leaf}  (this cell's one change from the family baseline)\n")
            for line in r["evidence"]:
                fh.write(f"===   [log]      {line}\n")
            fh.write(f"{'':=<118}\n{scrub(r['text']).rstrip()}\n")


soft = [r for r in hits if r["mode"] == "soft"]
hard = [r for r in hits if r["mode"] == "hard"]
write(SOFT_OUT, "soft", soft)
write(HARD_OUT, "hard", hard)
print(f"{len(soft)} soft -> {SOFT_OUT}")
print(f"{len(hard)} hard -> {HARD_OUT}")
