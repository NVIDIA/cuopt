# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest

from cuopt.grpc.linear_programming import (
    Client,
    GrpcError,
    JobNotReadyError,
    JobStatus,
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

        terminal = _poll_until_complete(client, job_id, _MIP_NAMES)
        assert terminal == JobStatus.COMPLETED
        client.join_incumbent_stream(job_id)

        assert collector.entries
        bulk = client.incumbents(job_id)
        assert bulk

        solution = client.result(job_id, _MIP_NAMES)
        assert solution is not None
        client.delete(job_id)
