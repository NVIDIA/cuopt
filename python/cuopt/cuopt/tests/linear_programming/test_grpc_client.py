# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import threading
import time

import pytest

from cuopt.grpc.client.grpc_client import (
    _WAIT_POLL_INTERVAL_S,
    _wait_poll_loop,
)
from cuopt.grpc.linear_programming import (
    Client,
    GrpcError,
    JobNotReadyError,
    JobStatus,
    TlsConfig,
)
from cuopt.linear_programming import Read, SolverSettings
from cuopt.linear_programming.internals import GetSolutionCallback
from cuopt.linear_programming.problem import INTEGER, MAXIMIZE, Problem
from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT

from grpc_server_fixtures import GRPC_PORT_OFFSET_CLIENT

RAPIDS_DATASET_ROOT_DIR = os.getenv("RAPIDS_DATASET_ROOT_DIR")
if RAPIDS_DATASET_ROOT_DIR is None:
    RAPIDS_DATASET_ROOT_DIR = os.getcwd()
    RAPIDS_DATASET_ROOT_DIR = os.path.join(RAPIDS_DATASET_ROOT_DIR, "datasets")

_SWATH1_MPS = os.path.join(RAPIDS_DATASET_ROOT_DIR, "mip", "swath1.mps")

_DEMO_LP_NAMES = ["x", "y"]
_MIP_NAMES = ["x", "y"]


def _demo_lp_problem():
    problem = Problem("grpc_demo")
    x = problem.addVariable(lb=0.0, ub=2.0, name="x")
    y = problem.addVariable(lb=0.0, name="y")
    problem.addConstraint(3 * x + 4 * y <= 5.4, name="c1")
    problem.addConstraint(2.7 * x + 10.1 * y <= 4.9, name="c2")
    problem.setObjective(0.2 * x + 0.1 * y, sense=MAXIMIZE)
    return problem


def _poll_until_complete(
    client, job_id, names, timeout=120, poll_interval=0.05
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.status(job_id)
        if status not in (JobStatus.QUEUED, JobStatus.PROCESSING):
            return status
        if client.result(job_id, names) is not None:
            return JobStatus.COMPLETED
        time.sleep(poll_interval)
    return client.status(job_id)


def _infeasible_lp_problem():
    problem = Problem("grpc_infeasible")
    x = problem.addVariable(lb=0.0, name="x")
    problem.addConstraint(x >= 5, name="c1")
    problem.addConstraint(x <= 1, name="c2")
    problem.setObjective(x, sense=MAXIMIZE)
    return problem


def _assert_demo_lp_solution(client):
    job_id = client.submit(_demo_lp_problem(), SolverSettings())
    try:
        assert client.wait(job_id, timeout=120) == JobStatus.COMPLETED
        solution = client.result(job_id, _DEMO_LP_NAMES)
        assert solution is not None
        assert solution.get_primal_objective() == pytest.approx(0.36, rel=1e-3)
    finally:
        client.delete(job_id)


class TestWaitPollLoop:
    def test_returns_immediately_when_already_terminal(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        status = _wait_poll_loop(
            lambda job_id: JobStatus.COMPLETED, "job", 0, GrpcError
        )
        assert status is JobStatus.COMPLETED
        assert sleeps == []

    def test_sleeps_between_in_flight_polls(self, monkeypatch):
        polls = []
        sleeps = []

        def get_status(job_id):
            polls.append(job_id)
            if len(polls) < 3:
                return JobStatus.PROCESSING
            return JobStatus.CANCELLED

        monkeypatch.setattr(time, "sleep", sleeps.append)
        status = _wait_poll_loop(get_status, "abc", 0, GrpcError)
        assert status is JobStatus.CANCELLED
        assert polls == ["abc", "abc", "abc"]
        assert sleeps == [_WAIT_POLL_INTERVAL_S, _WAIT_POLL_INTERVAL_S]

    def test_timeout_raises_after_deadline(self, monkeypatch):
        ticks = iter([100.0, 100.0, 101.0])
        monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
        with pytest.raises(
            GrpcError, match="Timeout waiting for job completion"
        ):
            _wait_poll_loop(
                lambda job_id: JobStatus.QUEUED, "job", 1, GrpcError
            )

    def test_rejects_negative_timeout(self):
        with pytest.raises(
            GrpcError, match="timeout_seconds must be non-negative"
        ):
            _wait_poll_loop(
                lambda job_id: JobStatus.PROCESSING, "job", -1, GrpcError
            )

    def test_other_thread_runs_during_sleep(self):
        started = threading.Event()
        progressed = threading.Event()
        polls = {"n": 0}

        def get_status(job_id):
            polls["n"] += 1
            if polls["n"] == 1:
                started.set()
                return JobStatus.PROCESSING
            assert progressed.is_set()
            return JobStatus.COMPLETED

        def worker():
            assert started.wait(timeout=2)
            progressed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        status = _wait_poll_loop(
            get_status,
            "job",
            0,
            GrpcError,
            poll_interval_s=0.05,
        )
        thread.join(timeout=2)
        assert status is JobStatus.COMPLETED
        assert progressed.is_set()

    def test_client_wait_releases_gil(self, monkeypatch):
        """A GIL-bound spinner must progress during Client.wait's poll sleep.

        Hits are counted only inside the sleep, so this fails if wait() still
        called the C++ WaitForCompletion/sleep path that holds the GIL.
        """
        import cuopt.grpc.client.grpc_client as grpc_mod

        polls = {"n": 0}

        def fake_status(self, job_id):
            polls["n"] += 1
            if polls["n"] == 1:
                return JobStatus.PROCESSING
            return JobStatus.COMPLETED

        monkeypatch.setattr(Client, "status", fake_status)
        client = Client.__new__(Client)

        hits = {"n": 0}
        during_sleep = []
        stop = threading.Event()

        def spinner():
            while not stop.is_set():
                hits["n"] += 1

        real_sleep = time.sleep

        def tracking_sleep(seconds):
            assert seconds == _WAIT_POLL_INTERVAL_S
            before = hits["n"]
            real_sleep(0.15)
            during_sleep.append(hits["n"] - before)

        monkeypatch.setattr(grpc_mod.time, "sleep", tracking_sleep)
        thread = threading.Thread(target=spinner)
        thread.start()
        try:
            status = client.wait("job")
        finally:
            stop.set()
            thread.join(timeout=2)

        assert status is JobStatus.COMPLETED
        assert during_sleep, "wait() never slept between status polls"
        assert during_sleep[0] > 1000, (
            "spinner made no progress during wait sleep; GIL likely held "
            f"(hits={during_sleep[0]})"
        )


class TestTlsConfig:
    def test_mtls_requires_both_client_materials(self):
        pem = "-----BEGIN CERTIFICATE-----\nabc\n-----END CERTIFICATE-----"
        with pytest.raises(ValueError, match="client_cert and client_key"):
            TlsConfig(pem, client_cert="client.crt")

    def test_accepts_pem_string(self):
        pem = "-----BEGIN CERTIFICATE-----\nabc\n-----END CERTIFICATE-----"
        cfg = TlsConfig(pem)
        assert cfg.root_certs == pem
        assert cfg.client_cert is None

    def test_accepts_none_root_certs(self):
        cfg = TlsConfig(root_certs=None)
        assert cfg.root_certs is None


@pytest.mark.xdist_group(name="grpc_server")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestGrpcClient:
    grpc_port_offset = GRPC_PORT_OFFSET_CLIENT
    grpc_server_yield = "port"

    def test_submit_status_result_delete(self, grpc_server):
        problem = _demo_lp_problem()
        settings = SolverSettings()
        client = Client("localhost", grpc_server)

        job_id = client.submit(problem, settings)
        assert job_id

        assert client.result(job_id, _DEMO_LP_NAMES) is None
        assert client.status(job_id) in (
            JobStatus.QUEUED,
            JobStatus.PROCESSING,
            JobStatus.COMPLETED,
        )

        terminal = client.wait(job_id, timeout=120)
        assert terminal == JobStatus.COMPLETED

        solution = client.result(job_id, _DEMO_LP_NAMES)
        assert solution is not None
        assert solution.get_primal_objective() == pytest.approx(0.36, rel=1e-3)
        vars_ = solution.get_vars()
        assert vars_["x"] == pytest.approx(1.8, rel=1e-3)
        assert vars_["y"] == pytest.approx(0.0, rel=1e-3)

        client.delete(job_id)

    def test_submit_with_log_stream(self, grpc_server):
        problem = _demo_lp_problem()
        settings = SolverSettings()

        client = Client("localhost", grpc_server)
        job_id = client.submit(problem, settings)

        received = []
        client.start_log_stream(job_id, callback=received.append)

        terminal = _poll_until_complete(client, job_id, _DEMO_LP_NAMES)
        assert terminal == JobStatus.COMPLETED
        state = client.join_log_stream(job_id)
        assert state is not None
        assert state["live_lines"] > 0, (
            "Log streaming failed; only backfill worked"
        )

        solution = client.result(job_id, _DEMO_LP_NAMES)
        assert solution is not None

        bulk_logs = client.logs(job_id)
        assert bulk_logs
        assert received
        assert len(received) == len(bulk_logs)

        client.delete(job_id)

    def test_logs_not_ready(self, grpc_server):
        problem = _demo_lp_problem()
        settings = SolverSettings()

        client = Client("localhost", grpc_server)
        job_id = client.submit(problem, settings)

        with pytest.raises(JobNotReadyError):
            client.logs(job_id)

        assert client.wait(job_id, timeout=120) == JobStatus.COMPLETED
        assert client.logs(job_id)

        client.delete(job_id)

    def test_mip_submit_and_result(self, grpc_server):
        problem = Problem("grpc_mip")
        x = problem.addVariable(lb=0, ub=10, vtype=INTEGER, name="x")
        y = problem.addVariable(lb=0, ub=10, vtype=INTEGER, name="y")
        problem.addConstraint(x + y <= 10, name="c1")
        problem.addConstraint(x - y >= 0, name="c2")
        problem.setObjective(x + 2 * y, sense=MAXIMIZE)

        client = Client("localhost", grpc_server)
        job_id = client.submit(problem, SolverSettings())
        assert client.wait(job_id, timeout=120) == JobStatus.COMPLETED

        solution = client.result(job_id, _MIP_NAMES)
        assert solution is not None
        assert solution.get_primal_objective() == pytest.approx(15.0, rel=1e-3)
        client.delete(job_id)

    def test_invalid_job_id(self, grpc_server):
        client = Client("localhost", grpc_server)
        assert (
            client.status("00000000-0000-0000-0000-000000000000")
            == JobStatus.NOT_FOUND
        )
        with pytest.raises(GrpcError):
            client.result("00000000-0000-0000-0000-000000000000")
        with pytest.raises(GrpcError):
            client.delete("00000000-0000-0000-0000-000000000000")

    def test_result_after_delete(self, grpc_server):
        problem = _demo_lp_problem()
        client = Client("localhost", grpc_server)
        job_id = client.submit(problem, SolverSettings())
        assert client.wait(job_id, timeout=120) == JobStatus.COMPLETED
        client.delete(job_id)
        with pytest.raises(GrpcError):
            client.result(job_id, _DEMO_LP_NAMES)

    def test_infeasible_lp_result(self, grpc_server):
        client = Client("localhost", grpc_server)
        job_id = client.submit(_infeasible_lp_problem(), SolverSettings())
        terminal = client.wait(job_id, timeout=120)
        if terminal != JobStatus.FAILED:
            client.delete(job_id)
            pytest.skip(
                f"expected FAILED for infeasible LP, got {terminal.name}"
            )
        with pytest.raises(GrpcError):
            client.result(job_id, ["x"])
        client.delete(job_id)

    def test_cancel_job(self, grpc_server):
        if not os.path.isfile(_SWATH1_MPS):
            pytest.skip(f"dataset not found: {_SWATH1_MPS}")

        problem = Read(_SWATH1_MPS)
        settings = SolverSettings()
        settings.set_parameter(CUOPT_TIME_LIMIT, 10)

        client = Client("localhost", grpc_server)
        job_id = client.submit(problem, settings)

        status = client.status(job_id)
        if status not in (JobStatus.QUEUED, JobStatus.PROCESSING):
            client.delete(job_id)
            pytest.skip("Job completed before cancellation could be observed")

        client.cancel(job_id)
        assert client.wait(job_id, timeout=30) == JobStatus.CANCELLED
        with pytest.raises(GrpcError):
            client.result(job_id)
        client.delete(job_id)

    def test_mip_incumbent_stream(self, grpc_server):
        class IncumbentCollector(GetSolutionCallback):
            def __init__(self):
                super().__init__()
                self.entries = []

            def get_solution(
                self, solution, solution_cost, solution_bound, user_data
            ):
                self.entries.append(
                    {
                        "solution": solution.tolist(),
                        "cost": float(solution_cost[0]),
                    }
                )

        problem = Problem("grpc_mip_incumbent")
        x = problem.addVariable(lb=0, ub=10, vtype=INTEGER, name="x")
        y = problem.addVariable(lb=0, ub=10, vtype=INTEGER, name="y")
        problem.addConstraint(x + y <= 10, name="c1")
        problem.addConstraint(x - y >= 0, name="c2")
        problem.setObjective(x + 2 * y, sense=MAXIMIZE)

        collector = IncumbentCollector()
        settings = SolverSettings()
        settings.set_mip_callback(collector, None)
        settings.set_parameter("time_limit", 30)

        client = Client("localhost", grpc_server)
        job_id = client.submit(problem, settings)
        client.start_incumbent_stream(job_id, settings=settings)

        terminal = client.wait(job_id, timeout=120)
        assert terminal == JobStatus.COMPLETED
        client.join_incumbent_stream(job_id)

        assert collector.entries
        bulk = client.incumbents(job_id)
        assert bulk

        solution = client.result(job_id, _MIP_NAMES)
        assert solution is not None
        client.delete(job_id)

    def test_mip_incumbent_stream_live_during_wait(self, grpc_server):
        """Incumbent callbacks must fire during wait(), not in a burst after.

        The 2-variable MIP in test_mip_incumbent_stream finishes too fast to
        tell. swath1 with a time limit stays PROCESSING long enough that a
        GIL-holding wait() would delay every callback until join().
        """
        if not os.path.isfile(_SWATH1_MPS):
            pytest.skip(f"dataset not found: {_SWATH1_MPS}")

        class TimedIncumbents(GetSolutionCallback):
            def __init__(self):
                super().__init__()
                self.times = []
                self.costs = []

            def get_solution(
                self, solution, solution_cost, solution_bound, user_data
            ):
                self.times.append(time.monotonic())
                self.costs.append(float(solution_cost[0]))

        collector = TimedIncumbents()
        settings = SolverSettings()
        settings.set_mip_callback(collector, None)
        settings.set_parameter(CUOPT_TIME_LIMIT, 8)

        client = Client("localhost", grpc_server)
        job_id = client.submit(Read(_SWATH1_MPS), settings)
        client.start_incumbent_stream(
            job_id, settings=settings, poll_interval_ms=200
        )
        try:
            terminal = client.wait(job_id, timeout=30)
            wait_end = time.monotonic()
            client.join_incumbent_stream(job_id)
        finally:
            client.delete(job_id)

        if terminal != JobStatus.COMPLETED:
            pytest.skip(f"job did not complete ({terminal.name})")
        if len(collector.times) < 2:
            pytest.skip(
                "need >=2 incumbents to test live delivery, got "
                f"{len(collector.times)}"
            )

        n_before = sum(t < wait_end for t in collector.times)
        spread = max(collector.times) - min(collector.times)
        lag = wait_end - min(collector.times)
        print(
            f"incumbents={len(collector.times)} before_wait={n_before} "
            f"spread={spread:.3f}s first_to_wait_end={lag:.3f}s"
        )
        assert n_before >= 1, (
            f"all {len(collector.times)} incumbents arrived at/after "
            f"wait() returned (spread={spread:.4f}s); GIL likely held"
        )
        assert spread > 0.15, (
            f"incumbent timestamps clustered in {spread:.4f}s "
            f"(n={len(collector.times)}, lag_to_wait_end={lag:.4f}s); "
            "likely dumped as a burst at completion"
        )


@pytest.mark.xdist_group(name="grpc_server")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestGrpcClientTls:
    def test_submit_with_explicit_tls_config(self, tls_server_info):
        cert_dir = tls_server_info["cert_dir"]
        client = Client(
            "localhost",
            tls_server_info["port"],
            tls=TlsConfig(os.path.join(cert_dir, "ca.crt")),
        )
        _assert_demo_lp_solution(client)

    def test_tls_server_rejects_plain_client(self, tls_server_info):
        with pytest.raises(GrpcError):
            Client("localhost", tls_server_info["port"], tls=False)

    def test_system_trust_rejects_self_signed_without_custom_ca(
        self, tls_server_info
    ):
        with pytest.raises(GrpcError):
            Client(
                "localhost",
                tls_server_info["port"],
                tls=TlsConfig(root_certs=None),
            )

    def test_submit_with_explicit_mtls_config(self, mtls_server_info):
        cert_dir = mtls_server_info["cert_dir"]
        client = Client(
            "localhost",
            mtls_server_info["port"],
            tls=TlsConfig(
                os.path.join(cert_dir, "ca.crt"),
                client_cert=os.path.join(cert_dir, "client.crt"),
                client_key=os.path.join(cert_dir, "client.key"),
            ),
        )
        _assert_demo_lp_solution(client)

    def test_mtls_server_rejects_missing_client_cert(self, mtls_server_info):
        cert_dir = mtls_server_info["cert_dir"]
        with pytest.raises(GrpcError):
            Client(
                "localhost",
                mtls_server_info["port"],
                tls=TlsConfig(os.path.join(cert_dir, "ca.crt")),
            )
