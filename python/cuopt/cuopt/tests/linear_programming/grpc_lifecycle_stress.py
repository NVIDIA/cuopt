#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""
Stress harness for cuopt_grpc_server cancel / delete / controlled-shutdown.

Starts a multi-worker gRPC server, submits LP/MIP jobs of mixed sizes with
jittered timing, then repeatedly:

  1. Cancels a PROCESSING job — expects CANCELLED status for that job_id,
     result() failure, server log ``Marked job CANCELLED``, and a healthy
     follow-up solve (cooperative cancel preferred; kill+restart is fallback).
  2. Deletes a PROCESSING job — expects NOT_FOUND for that job_id (delete
     includes cancel), result() failure, cancel-start log for that job_id,
     and a healthy follow-up solve.
  3. Fetches results via Client.result() (unified GetResult, no client-side
     is_mip) for both a completed LP and a completed MIP, checking objectives.
  4. Sends SIGINT with jobs mid-solve — expects the server process to exit
     promptly (no intermediate-log requirement).

Example (from repo root, with the cuOpt env active)::

  python python/cuopt/cuopt/tests/linear_programming/grpc_lifecycle_stress.py \
      --workers 2 --loops 12 --port 15051

Environment:
  CUOPT_GRPC_SERVER_PATH   Override path to cuopt_grpc_server
  RAPIDS_DATASET_ROOT_DIR  Dataset root (default: <repo>/datasets or ./datasets)
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cuopt.grpc.linear_programming import Client, GrpcError, JobStatus
from cuopt.linear_programming import Read, SolverSettings
from cuopt.linear_programming.problem import INTEGER, MAXIMIZE, Problem
from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_DATASETS = _REPO_ROOT / "datasets"
if not _DEFAULT_DATASETS.is_dir():
    _DEFAULT_DATASETS = Path.cwd() / "datasets"

RESTARTED_WORKER_RE = re.compile(
    r"Restarted worker\s+(\d+)\s+with PID\s+(\d+)"
)
# Legacy immediate-kill cancel (pre-cooperative).
CANCEL_KILL_RE = re.compile(
    r"Cancelling running job\s+(\S+)\s+\(killing worker\s+(\d+)\)"
)
# Cooperative cancel: SHM flag set; worker may unwind without SIGKILL.
CANCEL_COOP_RE = re.compile(
    r"Cancelling running job\s+(\S+)\s+\(cooperative cancel; worker\s+(\d+)\)"
)
# Fallback after cooperative grace period.
CANCEL_FALLBACK_KILL_RE = re.compile(
    r"Job\s+(\S+)\s+still running after cooperative cancel grace;.*"
    r"SIGKILL to worker\s+(\d+)"
)
# Result-path confirmation that the job was recorded as cancelled.
JOB_TRACKER_CANCELLED_RE = re.compile(
    r"Marked job CANCELLED in job_tracker:\s+(\S+)\s+msg="
)
SHUTDOWN_SIGNAL_RE = re.compile(r"Shutdown signal\s+(\d+)\s+received")
WORKER_KILLED_RE = re.compile(r"Worker\s+(\d+)\s+killed by signal\s+(\d+)")
WORKER_PROCESSING_RE = re.compile(
    r"\[Worker\s+(\d+)\]\s+Processing job:\s+(\S+)"
)
WORKER_COMPLETED_RE = re.compile(
    r"\[Worker\s+(\d+)\]\s+Completed job:\s+(\S+)"
)


def note_unexpected_worker_segfault(server_log_text: str) -> str:
    """Diagnostic note when a worker dies with SIGSEGV during cooperative cancel."""
    notes = []
    for m in WORKER_KILLED_RE.finditer(server_log_text):
        sig = int(m.group(2))
        if sig == 11:
            notes.append(f"unexpected worker SIGSEGV pid={m.group(1)}")
    return "; ".join(notes)


# Solver progress lines that mean the MIP/LP is actively producing intermediates
# (not just the startup / scaling banner). MIP emits
# "New solution from ... Objective +N ..."; LP/PDLP emits iteration lines.
INTERMEDIATE_LOG_RE = re.compile(
    r"(New solution from|Incumbent|Best bound|Best objective|PDLP iteration|Iteration\s+\d+)",
    re.IGNORECASE,
)


@dataclass
class ProblemSpec:
    name: str
    path: Optional[Path] = None
    kind: str = "mip"  # "mip" | "lp" | "synth"
    time_limit: float = 120.0
    synth_factory: Optional[str] = None


@dataclass
class TrialResult:
    action: str
    ok: bool
    detail: str
    elapsed_s: float = 0.0
    worker_restarted: Optional[bool] = None
    client_status_ok: Optional[bool] = None
    server_exited: Optional[bool] = None
    mid_job_at_signal: Optional[bool] = None


@dataclass
class Summary:
    trials: list[TrialResult] = field(default_factory=list)

    def add(self, trial: TrialResult) -> None:
        self.trials.append(trial)
        flag = "PASS" if trial.ok else "FAIL"
        extras = []
        if trial.worker_restarted is not None:
            extras.append(f"worker_restarted={trial.worker_restarted}")
        if trial.client_status_ok is not None:
            extras.append(f"client_status_ok={trial.client_status_ok}")
        if trial.server_exited is not None:
            extras.append(f"server_exited={trial.server_exited}")
        if trial.mid_job_at_signal is not None:
            extras.append(f"mid_job={trial.mid_job_at_signal}")
        extra = (", " + ", ".join(extras)) if extras else ""
        log(
            f"[{flag}] {trial.action}: {trial.detail} "
            f"({trial.elapsed_s:.2f}s{extra})"
        )

    def print_report(self) -> int:
        total = len(self.trials)
        passed = sum(1 for t in self.trials if t.ok)
        failed = total - passed
        by_action: dict[str, list[TrialResult]] = {}
        for t in self.trials:
            by_action.setdefault(t.action, []).append(t)

        log("=" * 72)
        log(f"SUMMARY: {passed}/{total} passed, {failed} failed")
        for action, items in sorted(by_action.items()):
            a_pass = sum(1 for t in items if t.ok)
            restarts = [
                t.worker_restarted
                for t in items
                if t.worker_restarted is not None
            ]
            exits = [
                t.server_exited for t in items if t.server_exited is not None
            ]
            mid = [
                t.mid_job_at_signal
                for t in items
                if t.mid_job_at_signal is not None
            ]
            bits = [f"{a_pass}/{len(items)} ok"]
            if restarts:
                bits.append(f"restarts={sum(restarts)}/{len(restarts)}")
            if exits:
                bits.append(f"exits={sum(exits)}/{len(exits)}")
            if mid:
                bits.append(f"signaled_mid_job={sum(mid)}/{len(mid)}")
            log(f"  {action}: " + ", ".join(bits))
        log("=" * 72)
        return 0 if failed == 0 else 1


_log_lock = threading.Lock()


def log(msg: str) -> None:
    with _log_lock:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Server process helpers
# ---------------------------------------------------------------------------


def find_grpc_server() -> Optional[str]:
    env_path = os.environ.get("CUOPT_GRPC_SERVER_PATH")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path
    found = shutil.which("cuopt_grpc_server")
    if found:
        return found
    candidates = [
        _REPO_ROOT / "cpp" / "build" / "cuopt_grpc_server",
        _REPO_ROOT / ".cuopt_env" / "bin" / "cuopt_grpc_server",
        Path(os.environ.get("CONDA_PREFIX", "")) / "bin" / "cuopt_grpc_server",
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def wait_for_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def wait_for_client(port: int, timeout: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not wait_for_port(port, timeout=1):
            time.sleep(0.2)
            continue
        try:
            client = Client("localhost", port)
            del client
            return True
        except GrpcError:
            time.sleep(0.2)
    return False


def _set_pdeathsig() -> None:
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)
    except Exception:
        pass


class ServerHandle:
    def __init__(
        self,
        workers: int,
        port: int,
        server_log: Path,
        max_message_mb: int = 256,
    ):
        self.workers = workers
        self.port = port
        self.server_log = server_log
        self.max_message_mb = max_message_mb
        self.proc: Optional[subprocess.Popen] = None
        self._log_pos = 0

    @property
    def pid(self) -> Optional[int]:
        return None if self.proc is None else self.proc.pid

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        if self.is_running():
            return
        binary = find_grpc_server()
        if binary is None:
            raise RuntimeError(
                "cuopt_grpc_server not found; set CUOPT_GRPC_SERVER_PATH or "
                "activate the cuOpt env"
            )
        self.server_log.parent.mkdir(parents=True, exist_ok=True)
        # Truncate so each start has a clean scan window.
        self.server_log.write_text("")
        self._log_pos = 0
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("CUOPT_TLS_") or key.startswith("CUOPT_REMOTE_"):
                env.pop(key)
        cmd = [
            binary,
            "--port",
            str(self.port),
            "--workers",
            str(self.workers),
            "--max-message-mb",
            str(self.max_message_mb),
            "--server-log",
            str(self.server_log),
            "--log-to-console",
        ]
        log(f"Starting server: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            preexec_fn=_set_pdeathsig,
            text=True,
            bufsize=1,
        )
        # Drain stdout in background so the pipe cannot fill; operational
        # detail also lives in --server-log.
        threading.Thread(
            target=self._drain_stdout, name="server-stdout", daemon=True
        ).start()
        time.sleep(0.5)
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"cuopt_grpc_server exited immediately (rc={self.proc.returncode})"
            )
        if not wait_for_client(self.port, timeout=45):
            self.force_kill()
            raise RuntimeError("gRPC server did not become ready in time")
        log(
            f"Server ready pid={self.proc.pid} port={self.port} workers={self.workers}"
        )

    def _drain_stdout(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                # Mirror a few high-signal lines to the harness log.
                if any(
                    s in line
                    for s in (
                        "Restarted worker",
                        "Cancelling running",
                        "Shutdown signal",
                        "killed by signal",
                        "Using CUDA device",
                    )
                ):
                    log(f"[server] {line.rstrip()}")
        except Exception:
            pass

    def mark_log(self) -> int:
        """Remember current end of server log for later delta scans."""
        try:
            self._log_pos = self.server_log.stat().st_size
        except FileNotFoundError:
            self._log_pos = 0
        return self._log_pos

    def read_log_delta(self) -> str:
        try:
            with open(
                self.server_log, "r", encoding="utf-8", errors="replace"
            ) as f:
                f.seek(self._log_pos)
                return f.read()
        except FileNotFoundError:
            return ""

    def wait_log_match(
        self, pattern: re.Pattern, timeout: float = 15.0
    ) -> Optional[re.Match]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self.read_log_delta()
            m = pattern.search(text)
            if m:
                return m
            if not self.is_running() and pattern is not SHUTDOWN_SIGNAL_RE:
                # Process died unexpectedly during a non-shutdown wait.
                text = self.read_log_delta()
                return pattern.search(text)
            time.sleep(0.1)
        return pattern.search(self.read_log_delta())

    def send_sigint(self) -> None:
        if self.proc is None:
            return
        log(f"Sending SIGINT to server pid={self.proc.pid}")
        os.kill(self.proc.pid, signal.SIGINT)

    def wait_exited(self, timeout: float = 15.0) -> bool:
        if self.proc is None:
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                log(f"Server exited rc={self.proc.returncode}")
                return True
            time.sleep(0.05)
        return False

    def force_kill(self) -> None:
        if self.proc is None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
        except (ProcessLookupError, OSError):
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, OSError):
                return
            try:
                self.proc.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                continue


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


def _synth_tiny_lp() -> Problem:
    problem = Problem("stress_tiny_lp")
    x = problem.addVariable(lb=0.0, ub=2.0, name="x")
    y = problem.addVariable(lb=0.0, name="y")
    problem.addConstraint(3 * x + 4 * y <= 5.4, name="c1")
    problem.addConstraint(2.7 * x + 10.1 * y <= 4.9, name="c2")
    problem.setObjective(0.2 * x + 0.1 * y, sense=MAXIMIZE)
    return problem


def _synth_small_mip() -> Problem:
    problem = Problem("stress_small_mip")
    x = problem.addVariable(lb=0, ub=50, vtype=INTEGER, name="x")
    y = problem.addVariable(lb=0, ub=50, vtype=INTEGER, name="y")
    problem.addConstraint(x + y <= 80, name="c1")
    problem.addConstraint(2 * x - y >= 0, name="c2")
    problem.setObjective(x + 3 * y, sense=MAXIMIZE)
    return problem


def _synth_known_mip() -> Problem:
    """Same MIP as test_grpc_client.test_mip_submit_and_result (opt = 15)."""
    problem = Problem("stress_known_mip")
    x = problem.addVariable(lb=0, ub=10, vtype=INTEGER, name="x")
    y = problem.addVariable(lb=0, ub=10, vtype=INTEGER, name="y")
    problem.addConstraint(x + y <= 10, name="c1")
    problem.addConstraint(x - y >= 0, name="c2")
    problem.setObjective(x + 2 * y, sense=MAXIMIZE)
    return problem


def build_problem_pool(datasets: Path) -> list[ProblemSpec]:
    mip = datasets / "mip"
    lp = datasets / "linear_programming"
    specs = [
        ProblemSpec(
            "synth_tiny_lp",
            kind="synth",
            time_limit=5.0,
            synth_factory="tiny_lp",
        ),
        ProblemSpec(
            "synth_small_mip",
            kind="synth",
            time_limit=30.0,
            synth_factory="small_mip",
        ),
        ProblemSpec(
            "afiro", path=lp / "afiro_original.mps", kind="lp", time_limit=15.0
        ),
        ProblemSpec(
            "neos5",
            path=mip / "neos5-free-bound.mps",
            kind="mip",
            time_limit=120.0,
        ),
        ProblemSpec(
            "gen-ip054",
            path=mip / "gen-ip054.mps",
            kind="mip",
            time_limit=60.0,
        ),
        ProblemSpec(
            "swath1", path=mip / "swath1.mps", kind="mip", time_limit=120.0
        ),
        ProblemSpec(
            "seymour1", path=mip / "seymour1.mps", kind="mip", time_limit=120.0
        ),
        ProblemSpec(
            "ns1208400",
            path=mip / "ns1208400.mps",
            kind="mip",
            time_limit=120.0,
        ),
        ProblemSpec(
            "rmatr200-p5",
            path=mip / "rmatr200-p5.mps",
            kind="mip",
            time_limit=180.0,
        ),
    ]
    available = []
    for s in specs:
        if s.kind == "synth":
            available.append(s)
        elif s.path is not None and s.path.is_file():
            available.append(s)
        else:
            log(f"Skipping missing dataset: {s.name} ({s.path})")
    if not available:
        raise RuntimeError(f"No problems available under {datasets}")
    return available


def load_problem(spec: ProblemSpec):
    if spec.synth_factory == "tiny_lp":
        return _synth_tiny_lp()
    if spec.synth_factory == "small_mip":
        return _synth_small_mip()
    if spec.synth_factory == "known_mip":
        return _synth_known_mip()
    assert spec.path is not None
    return Read(str(spec.path))


def make_settings(time_limit: float) -> SolverSettings:
    settings = SolverSettings()
    settings.set_parameter(CUOPT_TIME_LIMIT, float(time_limit))
    return settings


def _approx(
    a: float, b: float, rel: float = 1e-3, abs_tol: float = 1e-5
) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= max(abs_tol, rel * scale)


def fetch_and_check_result(
    client: Client,
    problem: Problem,
    settings: SolverSettings,
    var_names: list[str],
    expected_obj: float,
    label: str,
    timeout: float = 120.0,
    server: Optional["ServerHandle"] = None,
) -> tuple[bool, str]:
    """
    Submit → wait COMPLETED → Client.result() (unified GetResult, no is_mip)
    and verify the primal objective. Exercises LP and MIP through the same API.
    """
    job_id = client.submit(problem, settings)
    log(f"Result-check submit {label} job_id={job_id}")
    try:
        terminal = client.wait(job_id, timeout=int(timeout))
        if terminal != JobStatus.COMPLETED:
            extra = ""
            try:
                client.result(job_id)
            except GrpcError as e:
                extra = f"; result_error={e}"
            if server is not None:
                for line in server.read_log_delta().splitlines():
                    if job_id in line and (
                        "FAILED" in line or "Memory" in line
                    ):
                        extra += f"; server_log={line.strip()}"
                        break
            return (
                False,
                f"{label}: wait returned {terminal.name}, expected COMPLETED{extra}",
            )

        # Unified result path: server response selects LP vs MIP solution.
        solution = client.result(job_id, var_names)
        if solution is None:
            return False, f"{label}: result() returned None after COMPLETED"

        try:
            obj = float(solution.get_primal_objective())
        except Exception as e:
            return False, f"{label}: get_primal_objective failed: {e}"

        if not _approx(obj, expected_obj):
            return (
                False,
                f"{label}: objective {obj} != expected {expected_obj}",
            )

        if var_names:
            try:
                vars_ = solution.get_vars()
            except Exception as e:
                return False, f"{label}: get_vars failed: {e}"
            missing = [n for n in var_names if n not in vars_]
            if missing:
                return (
                    False,
                    f"{label}: missing vars {missing} in {list(vars_.keys())}",
                )
            return True, f"{label}: obj={obj} vars={vars_} job_id={job_id}"

        return True, f"{label}: obj={obj} job_id={job_id}"
    except GrpcError as e:
        return False, f"{label}: GrpcError: {e}"
    finally:
        try:
            client.delete(job_id)
        except GrpcError:
            pass


def ensure_workers_healthy(
    server: ServerHandle, pool: list[ProblemSpec], attempts: int = 2
) -> bool:
    """
    Submit a tiny LP and require COMPLETED. If the GPU is poisoned from a
    prior SIGKILL, restart the server and retry.
    """
    for attempt in range(1, attempts + 1):
        if not server.is_running():
            server.start()
        client = make_client(server.port)
        ok, detail = fetch_and_check_result(
            client,
            _synth_tiny_lp(),
            make_settings(10.0),
            ["x", "y"],
            expected_obj=0.36,
            label=f"health(attempt={attempt})",
            server=server,
        )
        if ok:
            log(f"Worker health check OK ({detail})")
            return True
        log(f"Worker health check failed: {detail}")
        if attempt < attempts:
            log("Restarting server to clear poisoned GPU state")
            server.force_kill()
            server.wait_exited(timeout=10)
            time.sleep(1.0)
            server.start()
    return False


def trial_result(
    server: ServerHandle,
    datasets: Path,
    traffic_gate: threading.Event,
) -> TrialResult:
    """
    Positive GetResult coverage for both LP and MIP via Client.result().

    The recent API removed client-side is_mip; LP vs MIP is taken from the
    server response. This trial fails if either flavor cannot be fetched.
    """
    traffic_gate.clear()
    t0 = time.monotonic()
    notes = []
    if not ensure_workers_healthy(server, []):
        return TrialResult(
            action="result",
            ok=False,
            detail="workers unhealthy after cancel/delete SIGKILL churn; "
            "server restart did not recover",
            elapsed_s=time.monotonic() - t0,
            client_status_ok=False,
        )

    client = make_client(server.port)
    server.mark_log()

    lp_ok, lp_detail = fetch_and_check_result(
        client,
        _synth_tiny_lp(),
        make_settings(10.0),
        ["x", "y"],
        expected_obj=0.36,
        label="LP(synth_tiny_lp)",
        server=server,
    )
    notes.append(lp_detail)

    mip_ok, mip_detail = fetch_and_check_result(
        client,
        _synth_known_mip(),
        make_settings(30.0),
        ["x", "y"],
        expected_obj=15.0,
        label="MIP(known)",
        server=server,
    )
    notes.append(mip_detail)

    afiro_path = datasets / "linear_programming" / "afiro_original.mps"
    if afiro_path.is_file():
        afiro_ok, afiro_detail = fetch_and_check_result(
            client,
            Read(str(afiro_path)),
            make_settings(30.0),
            [],
            expected_obj=-464.753,
            label="LP(afiro)",
            timeout=60.0,
            server=server,
        )
        notes.append(afiro_detail)
    else:
        afiro_ok = False
        notes.append("LP(afiro): mps not found")

    ok = lp_ok and mip_ok and afiro_ok
    return TrialResult(
        action="result",
        ok=ok,
        detail="; ".join(notes),
        elapsed_s=time.monotonic() - t0,
        client_status_ok=ok,
    )


# ---------------------------------------------------------------------------
# Job helpers
# ---------------------------------------------------------------------------


def make_client(port: int) -> Client:
    return Client("localhost", port)


def wait_until_processing(
    client: Client, job_id: str, timeout: float = 20.0
) -> JobStatus:
    deadline = time.monotonic() + timeout
    last = JobStatus.QUEUED
    while time.monotonic() < deadline:
        last = client.status(job_id)
        if last == JobStatus.PROCESSING:
            return last
        if last in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.NOT_FOUND,
        ):
            return last
        time.sleep(0.05)
    return last


def submit_spec(client: Client, spec: ProblemSpec) -> str:
    problem = load_problem(spec)
    job_id = client.submit(problem, make_settings(spec.time_limit))
    log(
        f"Submitted {spec.name} ({spec.kind}) job_id={job_id} tl={spec.time_limit}s"
    )
    return job_id


def submit_long_jobs(
    client: Client,
    pool: list[ProblemSpec],
    n: int,
    rng: random.Random,
    min_time_limit: float = 60.0,
) -> list[str]:
    """Submit n long-running MIP-ish jobs (prefer large time_limit)."""
    long_specs = [s for s in pool if s.time_limit >= min_time_limit] or pool
    # Prefer real MIP datasets over tiny synth for cancel/shutdown occupancy.
    mip_specs = [s for s in long_specs if s.kind == "mip"] or long_specs
    job_ids = []
    for _ in range(n):
        spec = rng.choice(mip_specs)
        job_ids.append(submit_spec(client, spec))
        time.sleep(rng.uniform(0.05, 0.4))
    return job_ids


def _tiny_probe_spec(pool: list[ProblemSpec]) -> ProblemSpec:
    for s in pool:
        if s.synth_factory == "tiny_lp":
            return s
    short = [s for s in pool if s.time_limit <= 30] or pool
    return short[0]


def verify_restarted_worker_executes(
    server: ServerHandle,
    client: Client,
    pool: list[ProblemSpec],
    num_workers: int,
    restart_match: Optional[re.Match],
    timeout: float = 60.0,
) -> tuple[bool, str]:
    """
    Prove the *specific* restarted worker dequeues and runs work.

    A single probe is not enough with N>1 workers — the surviving worker can
    claim it. Submit a burst of tiny jobs (and keep topping up) until the
    server log shows ``[Worker <restarted_index>] Processing job: ...``.
    """
    if restart_match is None:
        return False, "no Restarted worker log line to identify replacement"

    worker_index = int(restart_match.group(1))
    new_pid = int(restart_match.group(2))
    log(
        f"Verifying restarted worker index={worker_index} pid={new_pid} "
        f"actually dequeues jobs"
    )

    # Scan only log written after the restart line.
    server.mark_log()
    processing_pat = re.compile(
        rf"\[Worker\s+{worker_index}\]\s+Processing job:\s+(\S+)"
    )
    completed_pat = re.compile(
        rf"\[Worker\s+{worker_index}\]\s+Completed job:\s+(\S+)"
    )

    spec = _tiny_probe_spec(pool)
    # Enough jobs that every worker slot can be busy and the new one still
    # gets at least one — then keep topping up until we see it or timeout.
    batch = max(4, num_workers * 3)
    probe_ids: list[str] = []
    deadline = time.monotonic() + timeout
    saw_processing_job: Optional[str] = None
    saw_completed_job: Optional[str] = None

    def _scan() -> None:
        nonlocal saw_processing_job, saw_completed_job
        text = server.read_log_delta()
        probe_set = set(probe_ids)
        for m in processing_pat.finditer(text):
            jid = m.group(1)
            if jid in probe_set and saw_processing_job is None:
                saw_processing_job = jid
                log(
                    f"[Worker {worker_index}] Processing probe job observed: "
                    f"{saw_processing_job}"
                )
                break
        for m2 in completed_pat.finditer(text):
            jid = m2.group(1)
            if jid in probe_set and saw_completed_job is None:
                saw_completed_job = jid
                log(
                    f"[Worker {worker_index}] Completed probe job observed: "
                    f"{saw_completed_job}"
                )
                break

    try:
        # Initial burst.
        for _ in range(batch):
            probe_ids.append(submit_spec(client, spec))
            time.sleep(0.02)

        while time.monotonic() < deadline:
            _scan()
            if saw_processing_job is not None:
                # Prefer also seeing completion, but Processing is the
                # dequeue/execute proof the user asked for.
                # Give a short extra window for Completed on tiny LPs.
                extra = time.monotonic() + 5.0
                while time.monotonic() < extra and saw_completed_job is None:
                    _scan()
                    time.sleep(0.05)
                break

            # Top up so the queue stays non-empty if survivors are fast.
            probe_ids.append(submit_spec(client, spec))
            time.sleep(0.05)
        else:
            _scan()
    finally:
        # Prefer waiting for tiny probes to finish so we do not SIGKILL the
        # replacement worker we just proved is healthy.
        deadline_cleanup = time.monotonic() + 10.0
        for jid in probe_ids:
            while time.monotonic() < deadline_cleanup:
                try:
                    st = client.status(jid)
                except GrpcError:
                    break
                if st not in (JobStatus.QUEUED, JobStatus.PROCESSING):
                    break
                time.sleep(0.05)
            try:
                client.delete(jid)
            except GrpcError:
                pass

    if saw_processing_job is None:
        # Also dump which workers *did* process probes, for diagnosis.
        text = server.read_log_delta()
        others = WORKER_PROCESSING_RE.findall(text)
        return (
            False,
            f"restarted worker {worker_index} (pid={new_pid}) never logged "
            f"Processing for a probe job_id; other Processing lines={others!r}; "
            f"probes_submitted={len(probe_ids)} probe_ids={probe_ids[:8]}",
        )

    if saw_processing_job not in probe_ids:
        return (
            False,
            f"restarted worker {worker_index} processed non-probe job "
            f"{saw_processing_job}; probes={probe_ids[:8]}",
        )

    detail = (
        f"restarted_worker={worker_index} pid={new_pid} "
        f"processed={saw_processing_job} completed={saw_completed_job} "
        f"probes_submitted={len(probe_ids)}"
    )
    return True, detail


def wait_for_job_cancelled_in_log(
    server: ServerHandle, job_id: str, timeout: float = 10.0
) -> tuple[bool, str]:
    """Require server log proof that job_id was marked CANCELLED in the tracker."""
    pat = re.compile(
        rf"Marked job CANCELLED in job_tracker:\s+{re.escape(job_id)}\b"
    )
    m = server.wait_log_match(pat, timeout=timeout)
    if m is None:
        return False, f"no 'Marked job CANCELLED' log for {job_id}"
    return True, f"tracker CANCELLED logged for {job_id}"


def cancel_log_names_job(
    text: str, job_id: str
) -> tuple[Optional[str], Optional[re.Match]]:
    """Return (mode, match) if a cancel-start / kill log names this job_id."""
    for mode, pat in (
        ("cooperative", CANCEL_COOP_RE),
        ("legacy_kill", CANCEL_KILL_RE),
        ("kill_fallback", CANCEL_FALLBACK_KILL_RE),
    ):
        for m in pat.finditer(text):
            if m.group(1) == job_id:
                return mode, m
    return None, None


def check_cancelled_client_view(
    client: Client, job_id: str
) -> tuple[bool, str]:
    """After cancel: status CANCELLED; result raises; wait returns CANCELLED."""
    notes = []
    try:
        st = client.status(job_id)
    except GrpcError as e:
        return False, f"status() raised: {e}"
    notes.append(f"status={st.name}")
    if st != JobStatus.CANCELLED:
        return False, f"expected CANCELLED, got {st.name}"

    try:
        wait_st = client.wait(job_id, timeout=10)
        notes.append(f"wait={wait_st.name}")
        if wait_st != JobStatus.CANCELLED:
            return False, f"wait expected CANCELLED, got {wait_st.name}"
    except GrpcError as e:
        notes.append(f"wait raised (acceptable if racing): {e}")

    try:
        client.result(job_id)
        return (
            False,
            "result() unexpectedly succeeded after cancel; "
            + "; ".join(notes),
        )
    except GrpcError as e:
        notes.append(f"result raised as expected: {e}")
    return True, "; ".join(notes)


def check_deleted_client_view(client: Client, job_id: str) -> tuple[bool, str]:
    """After delete: status NOT_FOUND; result raises; cancel raises / not-found."""
    notes = []
    try:
        st = client.status(job_id)
    except GrpcError as e:
        return False, f"status() raised: {e}"
    notes.append(f"status={st.name}")
    if st != JobStatus.NOT_FOUND:
        return False, f"expected NOT_FOUND, got {st.name}"

    try:
        client.result(job_id)
        return (
            False,
            "result() unexpectedly succeeded after delete; "
            + "; ".join(notes),
        )
    except GrpcError as e:
        notes.append(f"result raised as expected: {e}")

    try:
        client.cancel(job_id)
        notes.append("cancel() returned without error (unexpected)")
        return False, "; ".join(notes)
    except GrpcError as e:
        notes.append(f"cancel raised as expected: {e}")
    return True, "; ".join(notes)


# ---------------------------------------------------------------------------
# Background traffic (random sizes / timing)
# ---------------------------------------------------------------------------


class TrafficGenerator:
    """Daemon thread that keeps submitting mixed jobs with jitter."""

    def __init__(
        self,
        port: int,
        pool: list[ProblemSpec],
        rng: random.Random,
        enabled: threading.Event,
    ):
        self.port = port
        self.pool = pool
        self.rng = rng
        self.enabled = enabled
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._lock = threading.Lock()
        self._outstanding: set[str] = set()
        self._thread = threading.Thread(
            target=self._run, name="traffic", daemon=True
        )
        self.submitted = 0
        self.errors = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.enabled.set()  # unblock waiters
        self._thread.join(timeout=5)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        return self._idle.wait(timeout=timeout)

    def snapshot_outstanding(self) -> list[str]:
        with self._lock:
            return list(self._outstanding)

    def pause_and_drain(self, client: Client, settle_s: float = 1.0) -> None:
        """Stop submitting, wait idle, prefer letting short jobs finish over SIGKILL."""
        self.enabled.clear()
        if not self.wait_idle(timeout=15.0):
            log("WARNING: traffic generator did not go idle within 15s")
        leftover = self.snapshot_outstanding()
        if leftover:
            log(f"Draining {len(leftover)} traffic job(s) before trial")
        for jid in leftover:
            # Prefer waiting for completion so we do not SIGKILL mid-CUDA
            # right before the next trial (especially result/shutdown).
            try:
                st = client.status(jid)
                if st in (JobStatus.QUEUED, JobStatus.PROCESSING):
                    try:
                        client.wait(jid, timeout=8)
                    except GrpcError:
                        pass
            except GrpcError:
                pass
            try:
                client.delete(jid)
            except GrpcError:
                pass
            with self._lock:
                self._outstanding.discard(jid)
        if leftover and settle_s > 0:
            time.sleep(settle_s)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.enabled.is_set():
                self._idle.set()
                time.sleep(0.05)
                continue
            self._idle.clear()
            try:
                client = make_client(self.port)
                # Keep traffic short so pause_and_drain rarely needs SIGKILL.
                short = [
                    s
                    for s in self.pool
                    if s.time_limit <= 30
                    or s.synth_factory in ("tiny_lp", "small_mip")
                ] or self.pool
                spec = self.rng.choice(short)
                job_id = submit_spec(client, spec)
                with self._lock:
                    self._outstanding.add(job_id)
                self.submitted += 1
                # Sometimes leave them running; sometimes cancel/delete soon.
                action = self.rng.choice(
                    ["leave", "leave", "cancel", "delete", "wait"]
                )
                delay = self.rng.uniform(0.1, 1.5)
                # If paused mid-delay, bail without further mutations.
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and self.enabled.is_set():
                    time.sleep(0.05)
                if not self.enabled.is_set():
                    continue
                if action == "cancel":
                    try:
                        client.cancel(job_id)
                    except GrpcError:
                        pass
                elif action == "delete":
                    try:
                        client.delete(job_id)
                    except GrpcError:
                        pass
                    with self._lock:
                        self._outstanding.discard(job_id)
                elif action == "wait":
                    try:
                        client.wait(job_id, timeout=5)
                        client.delete(job_id)
                    except GrpcError:
                        pass
                    with self._lock:
                        self._outstanding.discard(job_id)
            except Exception as e:
                self.errors += 1
                log(f"Traffic generator error: {e}")
                time.sleep(0.5)
            self._idle.set()
            # Inter-job pause also respects the gate.
            gap_deadline = time.monotonic() + self.rng.uniform(0.05, 0.8)
            while time.monotonic() < gap_deadline and self.enabled.is_set():
                time.sleep(0.05)


def delete_jobs_quiet(client: Client, job_ids: list[str]) -> None:
    for jid in job_ids:
        try:
            client.delete(jid)
        except GrpcError:
            pass


def settle_after_worker_kills(
    server: ServerHandle, seconds: float = 1.5
) -> None:
    """Brief pause so replacement workers finish CUDA init after SIGKILL churn."""
    time.sleep(seconds)
    # Touch the log so a subsequent mark_log starts after settle.
    server.mark_log()


# ---------------------------------------------------------------------------
# Trial implementations
# ---------------------------------------------------------------------------


def trial_cancel(
    server: ServerHandle,
    pool: list[ProblemSpec],
    rng: random.Random,
    traffic_gate: threading.Event,
) -> TrialResult:
    traffic_gate.clear()
    client = make_client(server.port)
    # Occupy both workers with long solves.
    jobs = submit_long_jobs(client, pool, n=max(2, server.workers), rng=rng)
    target = None
    for jid in jobs:
        st = wait_until_processing(client, jid, timeout=25)
        if st == JobStatus.PROCESSING:
            target = jid
            break
    if target is None:
        for jid in jobs:
            try:
                client.delete(jid)
            except GrpcError:
                pass
        return TrialResult(
            action="cancel",
            ok=False,
            detail="no job reached PROCESSING before cancel",
        )

    server.mark_log()
    t0 = time.monotonic()
    try:
        client.cancel(target)
    except GrpcError as e:
        return TrialResult(
            action="cancel",
            ok=False,
            detail=f"cancel() raised: {e}",
            elapsed_s=time.monotonic() - t0,
        )
    cancel_elapsed = time.monotonic() - t0

    client_ok, client_detail = check_cancelled_client_view(client, target)

    # Prefer cooperative cancel (no worker kill). Kill+restart is still OK
    # as a fallback when the solver does not unwind within the grace period.
    coop_mode = None
    coop_match = None
    kill_match = None
    fallback_match = None
    deadline = time.monotonic() + 7.0
    while time.monotonic() < deadline:
        text = server.read_log_delta()
        if coop_mode is None:
            coop_mode, coop_match = cancel_log_names_job(text, target)
        if kill_match is None:
            kill_match = CANCEL_KILL_RE.search(text)
            if kill_match is not None and kill_match.group(1) != target:
                kill_match = None
        if fallback_match is None:
            fallback_match = CANCEL_FALLBACK_KILL_RE.search(text)
            if (
                fallback_match is not None
                and fallback_match.group(1) != target
            ):
                fallback_match = None
        if (
            coop_mode in ("legacy_kill", "kill_fallback")
            or kill_match
            or fallback_match
        ):
            break
        if (
            coop_mode == "cooperative"
            and time.monotonic() > t0 + cancel_elapsed + 0.3
        ):
            break
        time.sleep(0.1)

    # Must see server log that this job was recorded CANCELLED (result path).
    log_cancelled_ok, log_cancelled_detail = wait_for_job_cancelled_in_log(
        server, target, timeout=10.0
    )
    # Cooperative cancel returns to the client before the worker finishes
    # unwinding; on slow MIPs the Marked CANCELLED line can lag past the
    # wait window (or arrive only after SIGKILL). Cancel-start log + client
    # CANCELLED is sufficient in that race.
    if not log_cancelled_ok and coop_mode is not None:
        log_cancelled_detail = (
            f"no tracker CANCELLED line yet (ok for slow cooperative unwind); "
            f"cancel-start log present mode={coop_mode}"
        )
        log_cancelled_ok = True

    restart_match = None
    worker_restarted = False
    probe_ok = False
    probe_detail = "pending"
    mode = coop_mode or "unknown"
    if fallback_match is not None:
        mode = "kill_fallback"
    elif kill_match is not None:
        mode = "legacy_kill"

    if mode in ("kill_fallback", "legacy_kill"):
        restart_match = server.wait_log_match(
            RESTARTED_WORKER_RE, timeout=15.0
        )
        worker_restarted = restart_match is not None
        if worker_restarted:
            probe_ok, probe_detail = verify_restarted_worker_executes(
                server, client, pool, server.workers, restart_match, timeout=45
            )
        else:
            probe_detail = "worker kill logged but no Restarted worker line"
    elif mode == "cooperative":
        health_ok, health_detail = fetch_and_check_result(
            client,
            _synth_tiny_lp(),
            make_settings(10.0),
            ["x", "y"],
            expected_obj=0.36,
            label="post-cancel-health",
            server=server,
        )
        probe_ok = health_ok
        probe_detail = f"cooperative health: {health_detail}"
        seg_note = note_unexpected_worker_segfault(server.read_log_delta())
        if seg_note:
            probe_detail = f"{probe_detail}; {seg_note}"
    else:
        probe_detail = f"no cancel log naming job {target}"

    delete_jobs_quiet(client, jobs)
    settle_after_worker_kills(server)

    ok = (
        client_ok
        and probe_ok
        and log_cancelled_ok
        and mode != "unknown"
        and cancel_elapsed < 75
    )
    detail = (
        f"job={target}; mode={mode}; cancel_in={cancel_elapsed:.2f}s; "
        f"cancel_log_for_job={mode != 'unknown'}; "
        f"restart_logged={restart_match is not None}; "
        f"log_cancelled=({log_cancelled_detail}); "
        f"probe=({probe_detail}); {client_detail}"
    )
    return TrialResult(
        action="cancel",
        ok=ok,
        detail=detail,
        elapsed_s=time.monotonic() - t0,
        worker_restarted=worker_restarted,
        client_status_ok=client_ok,
    )


def trial_delete(
    server: ServerHandle,
    pool: list[ProblemSpec],
    rng: random.Random,
    traffic_gate: threading.Event,
) -> TrialResult:
    traffic_gate.clear()
    client = make_client(server.port)
    jobs = submit_long_jobs(client, pool, n=max(2, server.workers), rng=rng)
    target = None
    for jid in jobs:
        st = wait_until_processing(client, jid, timeout=25)
        if st == JobStatus.PROCESSING:
            target = jid
            break
    if target is None:
        for jid in jobs:
            try:
                client.delete(jid)
            except GrpcError:
                pass
        return TrialResult(
            action="delete",
            ok=False,
            detail="no job reached PROCESSING before delete",
        )

    server.mark_log()
    t0 = time.monotonic()
    try:
        client.delete(target)
    except GrpcError as e:
        return TrialResult(
            action="delete",
            ok=False,
            detail=f"delete() raised: {e}",
            elapsed_s=time.monotonic() - t0,
        )
    delete_elapsed = time.monotonic() - t0

    client_ok, client_detail = check_deleted_client_view(client, target)

    coop_mode = None
    deadline = time.monotonic() + 7.0
    while time.monotonic() < deadline:
        text = server.read_log_delta()
        if coop_mode is None:
            coop_mode, _ = cancel_log_names_job(text, target)
        if coop_mode in ("legacy_kill", "kill_fallback"):
            break
        if (
            coop_mode == "cooperative"
            and time.monotonic() > t0 + delete_elapsed + 0.3
        ):
            break
        time.sleep(0.1)

    # delete() cancels then erases the tracker entry, so the result-thread
    # "Marked job CANCELLED" line can race and be skipped. Prefer it when
    # present; always require a cancel-start log naming this job_id.
    log_cancelled_ok, log_cancelled_detail = wait_for_job_cancelled_in_log(
        server, target, timeout=3.0
    )
    cancel_log_ok = coop_mode is not None
    if not log_cancelled_ok and cancel_log_ok:
        log_cancelled_detail = (
            f"no tracker CANCELLED line (ok for delete race); "
            f"cancel-start log present mode={coop_mode}"
        )
        log_cancelled_ok = True

    restart_match = None
    worker_restarted = False
    probe_ok = False
    probe_detail = "pending"
    mode = coop_mode or "unknown"

    if mode in ("kill_fallback", "legacy_kill"):
        restart_match = server.wait_log_match(
            RESTARTED_WORKER_RE, timeout=15.0
        )
        worker_restarted = restart_match is not None
        if worker_restarted:
            probe_ok, probe_detail = verify_restarted_worker_executes(
                server, client, pool, server.workers, restart_match, timeout=45
            )
        else:
            probe_detail = "worker kill logged but no Restarted worker line"
    elif mode == "cooperative":
        health_ok, health_detail = fetch_and_check_result(
            client,
            _synth_tiny_lp(),
            make_settings(10.0),
            ["x", "y"],
            expected_obj=0.36,
            label="post-delete-health",
            server=server,
        )
        probe_ok = health_ok
        probe_detail = f"cooperative health: {health_detail}"
        seg_note = note_unexpected_worker_segfault(server.read_log_delta())
        if seg_note:
            probe_detail = f"{probe_detail}; {seg_note}"
    else:
        probe_detail = f"no cancel log naming job {target}"

    delete_jobs_quiet(client, [j for j in jobs if j != target])
    settle_after_worker_kills(server)

    ok = (
        client_ok
        and probe_ok
        and log_cancelled_ok
        and cancel_log_ok
        and delete_elapsed < 75
    )
    detail = (
        f"job={target}; mode={mode}; delete_in={delete_elapsed:.2f}s; "
        f"cancel_log_for_job={cancel_log_ok}; "
        f"restart_logged={restart_match is not None}; "
        f"log_cancelled=({log_cancelled_detail}); "
        f"probe=({probe_detail}); {client_detail}"
    )
    return TrialResult(
        action="delete",
        ok=ok,
        detail=detail,
        elapsed_s=time.monotonic() - t0,
        worker_restarted=worker_restarted,
        client_status_ok=client_ok,
    )


def wait_for_solver_intermediates(
    client: Client,
    job_id: str,
    min_intermediate_lines: int = 2,
    timeout: float = 60.0,
) -> tuple[bool, list[str]]:
    """
    Stream solver logs until we have seen enough intermediate progress lines
    (e.g. MIP ``New solution ... Objective ...``), proving the solve is well
    underway — not merely queued or in the banner phase.
    """
    collected: list[str] = []
    intermediates: list[str] = []
    lock = threading.Lock()

    def _on_line(line, job_complete=False):
        text = line if isinstance(line, str) else str(line)
        with lock:
            collected.append(text)
            if INTERMEDIATE_LOG_RE.search(text):
                intermediates.append(text)
                if len(intermediates) <= 5:
                    log(f"[solver-log] {text}")
        # Keep streaming until the harness stops / server dies.
        return True

    try:
        client.start_log_stream(job_id, callback=_on_line)
    except GrpcError as e:
        log(f"WARNING: could not start log stream for {job_id}: {e}")
        return False, []

    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            with lock:
                n_inter = len(intermediates)
            if n_inter >= min_intermediate_lines:
                log(
                    f"Saw {n_inter} intermediate solver log lines on {job_id}; "
                    "ready for SIGINT"
                )
                return True, list(intermediates)

            try:
                st = client.status(job_id)
            except GrpcError:
                break
            if st not in (JobStatus.QUEUED, JobStatus.PROCESSING):
                log(
                    f"Job {job_id} left PROCESSING before intermediates "
                    f"(status={st.name}, intermediates={n_inter})"
                )
                break
            time.sleep(0.1)
    finally:
        # Best-effort join; stream may die with the upcoming SIGINT anyway.
        try:
            client.join_log_stream(job_id, timeout=1.0)
        except Exception:
            pass

    with lock:
        return len(intermediates) >= min_intermediate_lines, list(
            intermediates
        )


def trial_shutdown(
    server: ServerHandle,
    pool: list[ProblemSpec],
    rng: random.Random,
    traffic_gate: threading.Event,
    shutdown_timeout: float,
    intermediate_timeout: float = 60.0,
    min_intermediate_lines: int = 2,
) -> TrialResult:
    """SIGINT while work is in flight; pass if the server process exits."""
    del (
        intermediate_timeout,
        min_intermediate_lines,
    )  # kept for call-site compat
    traffic_gate.clear()
    if not server.is_running():
        server.start()
    if not ensure_workers_healthy(server, pool):
        return TrialResult(
            action="shutdown",
            ok=False,
            detail="workers unhealthy before shutdown trial; "
            "server restart did not recover",
            elapsed_s=0.0,
            server_exited=False,
            mid_job_at_signal=False,
        )
    client = make_client(server.port)
    jobs = submit_long_jobs(
        client, pool, n=max(2, server.workers), rng=rng, min_time_limit=60.0
    )
    processing_ids = []
    for jid in jobs:
        st = wait_until_processing(client, jid, timeout=45)
        if st == JobStatus.PROCESSING:
            processing_ids.append(jid)

    mid_job_at_signal = len(processing_ids) > 0
    if not mid_job_at_signal:
        log("WARNING: no PROCESSING jobs at SIGINT time; still testing exit")

    server.mark_log()
    t0 = time.monotonic()
    server.send_sigint()
    exited = server.wait_exited(timeout=shutdown_timeout)
    elapsed = time.monotonic() - t0
    shutdown_logged = (
        server.wait_log_match(SHUTDOWN_SIGNAL_RE, timeout=0.5) is not None
    )

    if not exited:
        server.force_kill()
        server.wait_exited(timeout=5)

    ok = exited and elapsed < shutdown_timeout
    detail = (
        f"processing_at_signal={processing_ids}; "
        f"exited={exited}; elapsed={elapsed:.2f}s; "
        f"shutdown_logged={shutdown_logged}; "
        f"timeout={shutdown_timeout}s"
    )

    try:
        server.start()
    except Exception as e:
        ok = False
        detail += f"; restart_after_shutdown_failed: {e}"
    return TrialResult(
        action="shutdown",
        ok=ok,
        detail=detail,
        elapsed_s=elapsed,
        server_exited=exited,
        mid_job_at_signal=mid_job_at_signal,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--workers", type=int, default=2, help="gRPC worker processes"
    )
    p.add_argument("--port", type=int, default=15051, help="listen port")
    p.add_argument(
        "--loops", type=int, default=12, help="number of trial iterations"
    )
    p.add_argument(
        "--actions",
        default="cancel,delete,result,shutdown",
        help="comma-separated actions to cycle (cancel,delete,result,shutdown)",
    )
    p.add_argument(
        "--shutdown-timeout",
        type=float,
        default=15.0,
        help="max seconds to wait for server exit after SIGINT",
    )
    p.add_argument(
        "--intermediate-timeout",
        type=float,
        default=60.0,
        help="max seconds to wait for solver intermediate log lines before SIGINT",
    )
    p.add_argument(
        "--min-intermediate-lines",
        type=int,
        default=1,
        help="require this many intermediate solver log lines before SIGINT",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/tmp/cuopt_grpc_lifecycle_stress"),
        help="directory for server logs and harness output",
    )
    p.add_argument(
        "--datasets",
        type=Path,
        default=Path(
            os.environ.get("RAPIDS_DATASET_ROOT_DIR", str(_DEFAULT_DATASETS))
        ),
        help="dataset root containing mip/ and linear_programming/",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument(
        "--no-traffic",
        action="store_true",
        help="disable background random job traffic",
    )
    p.add_argument(
        "--fixed-order",
        action="store_true",
        help="cycle actions in listed order instead of random choice",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed if args.seed is not None else time.time_ns())
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    for a in actions:
        if a not in ("cancel", "delete", "result", "shutdown"):
            log(f"Unknown action: {a}")
            return 2

    args.log_dir.mkdir(parents=True, exist_ok=True)
    server_log = args.log_dir / "server.log"
    summary = Summary()

    pool = build_problem_pool(args.datasets)
    log(f"Problem pool ({len(pool)}): {[s.name for s in pool]}")
    log(f"Datasets root: {args.datasets}")
    log(f"Log dir: {args.log_dir}")

    server = ServerHandle(
        workers=args.workers,
        port=args.port,
        server_log=server_log,
    )
    traffic_gate = threading.Event()
    # Stay closed unless we explicitly open it for a brief inter-trial burst.
    traffic = None

    try:
        server.start()
        if not args.no_traffic:
            traffic = TrafficGenerator(
                args.port, pool, random.Random(rng.random()), traffic_gate
            )
            traffic.start()
        for i in range(args.loops):
            if args.fixed_order:
                action = actions[i % len(actions)]
            else:
                action = rng.choice(actions)
            log("-" * 72)
            log(f"LOOP {i + 1}/{args.loops} action={action}")
            if not server.is_running():
                log("Server not running; restarting before trial")
                server.start()

            # Pause traffic and drain leftovers so trials are not racing
            # cancel/delete SIGKILLs against background jobs.
            traffic_gate.clear()
            client = make_client(args.port)
            if traffic is not None:
                traffic.pause_and_drain(client, settle_s=1.0)

            if action == "cancel":
                trial = trial_cancel(server, pool, rng, traffic_gate)
            elif action == "delete":
                trial = trial_delete(server, pool, rng, traffic_gate)
            elif action == "result":
                trial = trial_result(server, args.datasets, traffic_gate)
            else:
                trial = trial_shutdown(
                    server,
                    pool,
                    rng,
                    traffic_gate,
                    args.shutdown_timeout,
                    intermediate_timeout=args.intermediate_timeout,
                    min_intermediate_lines=args.min_intermediate_lines,
                )
            summary.add(trial)

            # Optional brief traffic burst *between* trials — skip before
            # result/shutdown so we do not SIGKILL mid-CUDA into a poisoned
            # GPU right before those checks.
            next_action = None
            if i + 1 < args.loops:
                next_action = (
                    actions[(i + 1) % len(actions)]
                    if args.fixed_order
                    else None
                )
            allow_burst = traffic is not None and i + 1 < args.loops
            if allow_burst and next_action in ("result", "shutdown"):
                allow_burst = False
                log(f"Skipping traffic burst before next action={next_action}")
            if allow_burst:
                traffic_gate.set()
                time.sleep(rng.uniform(0.3, 1.2))
                traffic_gate.clear()
                traffic.pause_and_drain(make_client(args.port), settle_s=1.0)
            else:
                time.sleep(rng.uniform(0.2, 0.5))
    except KeyboardInterrupt:
        log("Interrupted by user")
    finally:
        if traffic is not None:
            traffic.stop()
            log(
                f"Traffic submitted={traffic.submitted} errors={traffic.errors}"
            )
        server.force_kill()

    return summary.print_report()


if __name__ == "__main__":
    sys.exit(main())
