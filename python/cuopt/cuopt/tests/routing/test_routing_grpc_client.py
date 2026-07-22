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

    client = grpc_routing.RoutingClient(_SERVER)
    remote = client.solve(_small_vrp(), {"time_limit": 2.0})

    assert remote["status"] == 0, remote["status_message"]
    assert remote["vehicle_count"] >= 1
    assert remote["total_objective_value"] == pytest.approx(
        local.get_total_objective(), rel=0.2
    )


def test_submit_wait_result_lifecycle():
    client = grpc_routing.RoutingClient(_SERVER)
    job_id = client.submit(_small_vrp(), {"time_limit": 1.0})
    assert job_id
    client.wait(job_id, timeout=30)
    solution = client.result(job_id)
    assert "route" in solution
    client.delete(job_id)
