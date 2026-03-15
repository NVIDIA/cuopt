"""Benchmark runner and regression detector.

Discovers benchmark files under ``datasets/``, runs cuOpt against them before
and after code changes, and produces a ``RegressionReport``.

Timing strategy:
- ``benchmark_warmup_runs`` un-timed warm-up solves per instance (GPU warm-up)
- ``benchmark_timed_runs`` timed runs; median is used for comparison
- Geometric mean across instances for aggregate speed comparison
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class InstanceResult(BaseModel):
    path: str
    solve_time_s: float          # median timed run
    objective: float | None      # best objective value (None if unavailable)
    status: str                  # "ok" | "error" | "timeout"
    error_msg: str = ""


class BenchmarkRun(BaseModel):
    label: str                   # "baseline" | "candidate"
    results: list[InstanceResult] = field(default_factory=list)  # type: ignore[assignment]

    model_config = {"arbitrary_types_allowed": True}


@dataclass
class RegressionReport:
    baseline: BenchmarkRun
    candidate: BenchmarkRun
    speed_delta_pct: float       # positive = slower
    quality_delta_pct: float     # positive = worse objective
    speed_regression: bool
    quality_regression: bool
    speed_tolerance_pct: float
    quality_tolerance_pct: float

    def summary(self) -> str:
        lines = [
            f"Speed delta  : {self.speed_delta_pct:+.2f}% (threshold ±{self.speed_tolerance_pct}%)",
            f"Quality delta: {self.quality_delta_pct:+.2f}% (threshold ±{self.quality_tolerance_pct}%)",
            f"Speed regression  : {'YES — auto-rejecting' if self.speed_regression else 'No'}",
            f"Quality regression: {'YES — requires human review' if self.quality_regression else 'No'}",
        ]
        return "\n".join(lines)

    def per_instance_table(self) -> str:
        """Markdown table comparing baseline vs candidate per instance."""
        base_map = {r.path: r for r in self.baseline.results}
        rows = ["| Instance | Baseline (s) | Candidate (s) | Speed Δ% | Obj Δ% |",
                "|----------|-------------|--------------|---------|--------|"]
        for cr in self.candidate.results:
            br = base_map.get(cr.path)
            if br is None:
                continue
            speed_d = 100 * (cr.solve_time_s - br.solve_time_s) / max(br.solve_time_s, 1e-9)
            if br.objective is not None and cr.objective is not None and br.objective != 0:
                obj_d = 100 * (cr.objective - br.objective) / abs(br.objective)
            else:
                obj_d = 0.0
            rows.append(
                f"| {Path(cr.path).name} | {br.solve_time_s:.4f} | "
                f"{cr.solve_time_s:.4f} | {speed_d:+.2f}% | {obj_d:+.2f}% |"
            )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Benchmark discovery
# ---------------------------------------------------------------------------

def _discover_instances(repo_root: Path, config: dict[str, Any]) -> list[Path]:
    datasets_root = repo_root / config.get("datasets_root", "datasets")
    exts = set(config.get("supported_extensions", [".mps", ".json", ".qps"]))
    instances: list[Path] = []
    if datasets_root.is_dir():
        for p in sorted(datasets_root.rglob("*")):
            if p.is_file() and p.suffix in exts:
                instances.append(p)
    return instances


# ---------------------------------------------------------------------------
# Single-instance runner
# ---------------------------------------------------------------------------

def _run_instance(
    instance: Path,
    repo_root: Path,
    warmup_runs: int,
    timed_runs: int,
) -> InstanceResult:
    """Run cuOpt on a single dataset file and return timing + objective."""

    # Determine runner based on file type
    suffix = instance.suffix.lower()
    if suffix == ".mps":
        cmd_base = ["python", "-m", "cuopt", "solve", "--mps", str(instance)]
    elif suffix == ".qps":
        cmd_base = ["python", "-m", "cuopt", "solve", "--qps", str(instance)]
    elif suffix == ".json":
        cmd_base = ["python", "-m", "cuopt", "solve", "--json", str(instance)]
    else:
        return InstanceResult(
            path=str(instance), solve_time_s=0, objective=None, status="error",
            error_msg=f"Unsupported extension: {suffix}"
        )

    env = {**os.environ, "CUOPT_BENCH_MODE": "1"}

    # Warm-up runs (discarded)
    for _ in range(warmup_runs):
        try:
            subprocess.run(cmd_base, capture_output=True, timeout=120, env=env, cwd=repo_root)
        except subprocess.TimeoutExpired:
            pass

    # Timed runs
    times: list[float] = []
    last_output = ""
    for _ in range(timed_runs):
        try:
            t0 = time.perf_counter()
            result = subprocess.run(
                cmd_base, capture_output=True, text=True, timeout=300, env=env, cwd=repo_root
            )
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            last_output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return InstanceResult(
                path=str(instance), solve_time_s=999.0, objective=None,
                status="timeout", error_msg="Timed out after 300s"
            )
        except Exception as exc:
            return InstanceResult(
                path=str(instance), solve_time_s=0, objective=None,
                status="error", error_msg=str(exc)
            )

    if not times:
        return InstanceResult(
            path=str(instance), solve_time_s=0, objective=None, status="error",
            error_msg="No timed runs completed"
        )

    median_time = sorted(times)[len(times) // 2]
    objective = _extract_objective(last_output)

    return InstanceResult(
        path=str(instance),
        solve_time_s=median_time,
        objective=objective,
        status="ok",
    )


def _extract_objective(output: str) -> float | None:
    """Parse the best objective value from cuOpt stdout."""
    for line in output.splitlines():
        lower = line.lower()
        for kw in ("objective:", "obj:", "best bound:", "optimal value:"):
            if kw in lower:
                parts = line.split()
                for part in reversed(parts):
                    try:
                        return float(part.replace(",", ""))
                    except ValueError:
                        continue
    return None


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    label: str,
    repo_root: Path,
    config: dict[str, Any],
    max_instances: int = 30,
) -> BenchmarkRun:
    """Run the full benchmark suite and return a ``BenchmarkRun``."""
    instances = _discover_instances(repo_root, config)[:max_instances]
    warmup = config.get("benchmark_warmup_runs", 2)
    timed = config.get("benchmark_timed_runs", 5)

    results: list[InstanceResult] = []
    for inst in instances:
        logger.info("[%s] Benchmarking %s ...", label, inst.name)
        r = _run_instance(inst, repo_root, warmup, timed)
        results.append(r)
        logger.debug("  → %.4fs  obj=%s  status=%s", r.solve_time_s, r.objective, r.status)

    run = BenchmarkRun(label=label, results=results)
    return run


# ---------------------------------------------------------------------------
# Regression comparison
# ---------------------------------------------------------------------------

def _geomean(values: list[float]) -> float:
    if not values:
        return 0.0
    log_sum = sum(math.log(max(v, 1e-12)) for v in values)
    return math.exp(log_sum / len(values))


def compare_runs(
    baseline: BenchmarkRun,
    candidate: BenchmarkRun,
    speed_tolerance_pct: float,
    quality_tolerance_pct: float,
) -> RegressionReport:
    """Compare two benchmark runs and produce a regression report."""
    base_map = {r.path: r for r in baseline.results}

    speed_ratios: list[float] = []
    quality_deltas: list[float] = []

    for cr in candidate.results:
        br = base_map.get(cr.path)
        if br is None or br.status != "ok" or cr.status != "ok":
            continue
        speed_ratios.append(cr.solve_time_s / max(br.solve_time_s, 1e-12))
        if br.objective is not None and cr.objective is not None and br.objective != 0:
            quality_deltas.append(
                100 * (cr.objective - br.objective) / abs(br.objective)
            )

    speed_delta_pct = 100 * (_geomean(speed_ratios) - 1.0) if speed_ratios else 0.0
    quality_delta_pct = max(quality_deltas) if quality_deltas else 0.0

    return RegressionReport(
        baseline=baseline,
        candidate=candidate,
        speed_delta_pct=speed_delta_pct,
        quality_delta_pct=quality_delta_pct,
        speed_regression=speed_delta_pct > speed_tolerance_pct,
        quality_regression=quality_delta_pct > quality_tolerance_pct,
        speed_tolerance_pct=speed_tolerance_pct,
        quality_tolerance_pct=quality_tolerance_pct,
    )
