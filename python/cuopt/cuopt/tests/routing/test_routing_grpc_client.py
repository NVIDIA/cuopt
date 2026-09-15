# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the compiled VRP gRPC client (cuopt.grpc.routing).

Skipped unless ``CUOPT_GRPC_SERVER`` points at a running cuopt_grpc_server that
supports VRP (``host:port``). Run locally with, e.g.::

    cuopt_grpc_server --port 50051 &
    CUOPT_GRPC_SERVER=localhost:50051 pytest test_grpc_client.py
"""

import os

import numpy as np
import pytest

from cuopt import routing

grpc_routing = pytest.importorskip("cuopt.grpc.routing")

_SERVER = os.environ.get("CUOPT_GRPC_SERVER")
pytestmark = pytest.mark.skipif(
    not _SERVER, reason="set CUOPT_GRPC_SERVER=host:port to run VRP gRPC tests"
)


def _client():
    host, _, port = (_SERVER or "").rpartition(":")
    return grpc_routing.RoutingClient(host, int(port))


def _small_vrp():
    dm = routing.DataModel(5, 2)
    cost = np.array(
        [
            [0, 1, 2, 2, 1],
            [1, 0, 1, 2, 2],
            [2, 1, 0, 1, 2],
            [2, 2, 1, 0, 1],
            [1, 2, 2, 1, 0],
        ],
        dtype=np.float32,
    )
    dm.add_cost_matrix(cost)
    return dm


def test_remote_solve_matches_local():
    settings = routing.SolverSettings()
    settings.set_time_limit(2)
    local = routing.Solve(_small_vrp(), settings)

    client = _client()
    remote = client.solve(_small_vrp(), {"time_limit": 2.0})

    assert remote["status"] == 0, remote["status_message"]
    assert remote["vehicle_count"] >= 1
    assert remote["total_objective_value"] == pytest.approx(
        local.get_total_objective(), rel=0.2
    )


def test_submit_wait_result_lifecycle():
    client = _client()
    job_id = client.submit(_small_vrp(), {"time_limit": 1.0})
    assert job_id
    assert client.status(job_id) in (
        grpc_routing.JobStatus.QUEUED,
        grpc_routing.JobStatus.PROCESSING,
        grpc_routing.JobStatus.COMPLETED,
    )
    assert client.wait(job_id, timeout=30) == grpc_routing.JobStatus.COMPLETED
    solution = client.result(job_id)
    assert "route" in solution
    client.delete(job_id)


def test_cancel_job():
    client = _client()
    job_id = client.submit(_small_vrp(), {"time_limit": 10.0})
    status = client.status(job_id)
    if status not in (
        grpc_routing.JobStatus.QUEUED,
        grpc_routing.JobStatus.PROCESSING,
    ):
        client.delete(job_id)
        pytest.skip("Job completed before cancellation could be observed")

    client.cancel(job_id)
    assert client.wait(job_id, timeout=30) == grpc_routing.JobStatus.CANCELLED
    client.delete(job_id)


def test_invalid_job_id():
    client = _client()
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.status(missing) == grpc_routing.JobStatus.NOT_FOUND
    with pytest.raises(grpc_routing.RoutingSolveError):
        client.cancel(missing)
