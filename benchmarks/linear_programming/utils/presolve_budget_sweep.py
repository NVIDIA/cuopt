# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Sweep the MIP presolve budget policies over a set of instances.

Runs cuopt_cli once per (instance, policy), scrapes the PRESOLVE_* log lines plus the final result
line, and writes one CSV row per run. The CSV carries both the structural features each budget was
derived from and what the budget actually spent, which is what a fit of
"budget <- problem dimensions and structure" needs.

Sweeping whole policies::

    python presolve_budget_sweep.py --cli ./cpp/build/cuopt_cli \
        --dataset-dir datasets/mip/miplib2017 --time-limit 300 --policies 0 2 \
        --out /tmp/presolve_sweep.csv --log-dir /tmp/presolve_sweep_logs

Sweeping one knob instead gives the mapping curve from a wall-clock limit onto a
round / badge / work-unit budget::

    python presolve_budget_sweep.py --grid-param mip_hyper_heuristic_presolve_max_rounds \
        --grid-values 1 2 3 5 10 20 30 50 --time-limit 2 --out /tmp/rounds_map.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

# Chosen to span the structural axes the policies key on: nnz, average row length, and binary
# fraction. bab2 / supportcase6 / square41 are the instances whose presolve ran away unbounded and
# motivated the budgets in the first place, so they are the regression cases.
DEFAULT_INSTANCES = [
    "bab2",
    "supportcase6",
    "square41",
    "air05",
    "30n20b8",
    "gen-ip054",
    "nw04",
    "rail507",
    "seymour",
    "mzzv11",
    "roll3000",
    "ns1208400",
    "glass4",
    "timtab1",
    "enlight_hard",
    "sp97ar",
]

POLICY_NAMES = {
    0: "legacy",
    1: "fixed",
    2: "size",
    3: "density",
    4: "binary",
    5: "combined",
    6: "manual",
}

KV_RE = re.compile(r"(\w+)=(-?[\w.+-]+)")
EXPLORED_RE = re.compile(
    r"Explored (\d+) nodes \((\d+) simplex iterations\) in ([\d.]+)s"
)
OBJ_RE = re.compile(
    r"Best objective ([-\d.eE+]+), best bound ([-\d.eE+]+), gap ([-\d.eE+]+|inf)%"
)

STATUS_MARKERS = [
    ("Optimal solution found", "Optimal"),
    ("Time limit reached", "TimeLimit"),
    ("Work limit reached", "WorkLimit"),
    ("Problem is infeasible", "Infeasible"),
    ("Problem is unbounded", "Unbounded"),
    ("No solution found", "NoSolution"),
    # A claim of integer infeasibility on an instance known to be feasible means a reduction was
    # unsound, so it must be distinguishable from simply not finding a solution in time.
    ("Problem has no integer feasible solution", "NoIntegerFeasible"),
]


def parse_kv(line, prefix):
    """Pull every key=value token that follows `prefix` on `line`."""
    idx = line.find(prefix)
    if idx < 0:
        return {}
    return dict(KV_RE.findall(line[idx + len(prefix) :]))


def parse_log(text):
    row = {}
    for line in text.splitlines():
        if "PRESOLVE_BUDGET stage=PAPILO" in line:
            for k, v in parse_kv(line, "PRESOLVE_BUDGET").items():
                row[f"papilo_{k}"] = v
        elif "PRESOLVE_BUDGET stage=PROBING" in line:
            for k, v in parse_kv(line, "PRESOLVE_BUDGET").items():
                row[f"probing_{k}"] = v
        elif "PRESOLVE_PAPILO_REDUCED" in line:
            for k, v in parse_kv(line, "PRESOLVE_PAPILO_REDUCED").items():
                row[f"reduced_{k}"] = v
        elif "PRESOLVE_PAPILO wall=" in line:
            for k, v in parse_kv(line, "PRESOLVE_PAPILO").items():
                row["papilo_wall" if k == "wall" else f"papilo_{k}"] = v
        elif "PRESOLVE_PROBING_WALL" in line:
            row["probing_wall"] = parse_kv(line, "PRESOLVE_PROBING_WALL").get(
                "wall"
            )
        elif "PRESOLVE_PROBING probes=" in line:
            for k, v in parse_kv(line, "PRESOLVE_PROBING").items():
                row[f"spent_{k}"] = v
        elif "Probing-cache step disabled" in line:
            row["probing_disabled"] = "1"

        m = EXPLORED_RE.search(line)
        if m:
            row["nodes"], row["simplex_iters"], row["solve_wall"] = m.groups()
        m = OBJ_RE.search(line)
        if m:
            row["objective"], row["bound"], row["gap_pct"] = m.groups()
        for marker, status in STATUS_MARKERS:
            if marker in line:
                row["status"] = status
    return row


def as_text(stream):
    """Decode a stream, since subprocess hands back bytes on the timeout path even with text=True."""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", "replace")
    return stream


def run_one(args, instance, policy, config_path, grid_value=None):
    with open(config_path, "w") as fh:
        fh.write(f"mip_hyper_heuristic_presolve_budget_policy = {policy}\n")
        if grid_value is not None:
            fh.write(f"{args.grid_param} = {grid_value}\n")
        for extra in args.param:
            fh.write(extra.replace(":", " = ", 1) + "\n")

    cmd = [
        args.cli,
        os.path.join(args.dataset_dir, instance + ".mps"),
        "--time-limit",
        str(args.time_limit),
        "--params-file",
        config_path,
    ]
    if args.determinism:
        cmd += ["--mip-determinism-mode", "1"]

    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        output = proc.stdout + proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = as_text(exc.stdout) + as_text(exc.stderr)
        returncode = -1
        timed_out = True
    wall = time.time() - t0

    row = parse_log(output)
    row.update(
        instance=instance,
        policy=policy,
        policy_name=POLICY_NAMES.get(policy, str(policy)),
        harness_wall=f"{wall:.2f}",
        harness_timeout=int(timed_out),
        returncode=returncode,
    )
    if grid_value is not None:
        row["grid_param"] = args.grid_param
        row["grid_value"] = grid_value
    if timed_out:
        row.setdefault("status", "HarnessTimeout")

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(args.log_dir, f"{instance}.p{policy}.log"), "w"
        ) as fh:
            fh.write(output)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="./cpp/build/cuopt_cli")
    ap.add_argument("--dataset-dir", default="datasets/mip/miplib2017")
    ap.add_argument("--instances", nargs="*", default=DEFAULT_INSTANCES)
    ap.add_argument(
        "--policies", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5]
    )
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="hard wall cap per run; the legacy policy leaves presolve unbounded",
    )
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        help="extra config entry as key:value, repeatable",
    )
    ap.add_argument(
        "--grid-param",
        default="",
        help="sweep this single hyper-parameter under the manual policy instead of sweeping "
        "policies; this is what yields the wall-limit -> rounds / work-unit mapping curve",
    )
    ap.add_argument("--grid-values", nargs="*", default=[])
    ap.add_argument("--out", default="presolve_sweep.csv")
    ap.add_argument("--log-dir", default="")
    args = ap.parse_args()

    config_path = "/tmp/presolve_budget_sweep.config"
    rows = []
    # A grid sweeps one knob under the manual policy; otherwise the variant axis is the policy.
    if args.grid_param:
        variants = [(6, v) for v in args.grid_values]
    else:
        variants = [(p, None) for p in args.policies]

    total = len(args.instances) * len(variants)
    done = 0
    for instance in args.instances:
        path = os.path.join(args.dataset_dir, instance + ".mps")
        if not os.path.exists(path):
            print(f"SKIP missing {path}", flush=True)
            continue
        for policy, grid_value in variants:
            done += 1
            # A sweep is long enough that losing all of it to one bad run is the worst outcome.
            try:
                row = run_one(args, instance, policy, config_path, grid_value)
            except Exception as exc:  # noqa: BLE001
                row = {
                    "instance": instance,
                    "policy": policy,
                    "policy_name": POLICY_NAMES.get(policy, str(policy)),
                    "grid_value": grid_value,
                    "status": "HarnessError",
                    "harness_wall": "0",
                    "harness_error": repr(exc),
                }
            rows.append(row)
            variant = (
                f"{args.grid_param.split('_')[-1]}={grid_value}"
                if grid_value is not None
                else f"p{policy} {row.get('policy_name', '')}"
            )
            print(
                f"[{done}/{total}] {instance:16s} {variant:14s}"
                f" status={row.get('status', '?'):18s}"
                f" papilo_wall={row.get('papilo_wall', '-'):>8s}"
                f" red_vars={row.get('reduced_nvars', '-'):>8s}"
                f" red_nnz={row.get('reduced_nnz', '-'):>9s}"
                f" probing_wall={row.get('probing_wall', '-'):>8s}"
                f" probes={row.get('spent_probes', '-'):>7s}"
                f" work={row.get('spent_work', '-'):>9s}",
                flush=True,
            )
            # Written incrementally so a long sweep is inspectable while it runs.
            write_csv(args.out, rows)

    write_csv(args.out, rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    lead = [
        "instance",
        "policy",
        "policy_name",
        "grid_param",
        "grid_value",
        "status",
    ]
    fields = lead + [f for f in fields if f not in lead]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
