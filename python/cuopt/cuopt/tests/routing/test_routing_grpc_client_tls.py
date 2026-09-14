# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TLS/mTLS coverage for the compiled VRP gRPC client (cuopt.grpc.routing).

Unlike test_routing_grpc_client.py, these tests do not need
CUOPT_GRPC_SERVER -- the tls_server_info/mtls_server_info fixtures start
their own cuopt_grpc_server subprocess (skipped if the test TLS certs are
not found).
"""

import os

import numpy as np
import pytest

from cuopt import routing
from cuopt.grpc.linear_programming import TlsConfig

grpc_routing = pytest.importorskip("cuopt.grpc.routing")


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


def test_rejects_invalid_tls_argument():
    with pytest.raises(TypeError):
        grpc_routing.RoutingClient("localhost", 1, tls="bogus")


@pytest.mark.xdist_group(name="grpc_server")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestRoutingClientTls:
    def test_submit_with_explicit_tls_config(self, tls_server_info):
        cert_dir = tls_server_info["cert_dir"]
        client = grpc_routing.RoutingClient(
            "localhost",
            tls_server_info["port"],
            tls=TlsConfig(os.path.join(cert_dir, "ca.crt")),
        )
        solution = client.solve(_small_vrp(), {"time_limit": 2.0})
        assert solution["status"] == 0, solution["status_message"]

    def test_tls_server_rejects_plain_client(self, tls_server_info):
        with pytest.raises(grpc_routing.RoutingSolveError):
            grpc_routing.RoutingClient(
                "localhost", tls_server_info["port"], tls=False
            )

    def test_submit_with_explicit_mtls_config(self, mtls_server_info):
        cert_dir = mtls_server_info["cert_dir"]
        client = grpc_routing.RoutingClient(
            "localhost",
            mtls_server_info["port"],
            tls=TlsConfig(
                os.path.join(cert_dir, "ca.crt"),
                client_cert=os.path.join(cert_dir, "client.crt"),
                client_key=os.path.join(cert_dir, "client.key"),
            ),
        )
        solution = client.solve(_small_vrp(), {"time_limit": 2.0})
        assert solution["status"] == 0, solution["status_message"]

    def test_mtls_server_rejects_missing_client_cert(self, mtls_server_info):
        cert_dir = mtls_server_info["cert_dir"]
        with pytest.raises(grpc_routing.RoutingSolveError):
            grpc_routing.RoutingClient(
                "localhost",
                mtls_server_info["port"],
                tls=TlsConfig(os.path.join(cert_dir, "ca.crt")),
            )
