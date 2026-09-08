# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal VRP demo for the NVIDIA cuOpt VRP gRPC client.

Unlike LP/MIP, routing has no ``CUOPT_REMOTE_HOST``/``CUOPT_REMOTE_PORT``
transparent path yet -- build a :class:`cuopt.routing.DataModel` and solve it
with :class:`cuopt.grpc.routing.RoutingClient`, an explicit client (host and
port passed directly).

Start the server first::

    cuopt_grpc_server --port 5001 --workers 1

Then::

    python remote_routing_demo.py
"""

import numpy as np
from cuopt import routing
from cuopt.grpc.routing import RoutingClient

dm = routing.DataModel(5, 2)
cost_matrix = np.array(
    [
        [0, 1, 2, 2, 1],
        [1, 0, 1, 2, 2],
        [2, 1, 0, 1, 2],
        [2, 2, 1, 0, 1],
        [1, 2, 2, 1, 0],
    ],
    dtype=np.float32,
)
dm.add_cost_matrix(cost_matrix)

client = RoutingClient("localhost", 5001)  # tls=None uses CUOPT_TLS_* if set
solution = client.solve(dm, {"time_limit": 5.0})

print("Status:    ", solution["status_message"])
print("Vehicles:  ", solution["vehicle_count"])
print("Objective: ", solution["total_objective_value"])
print("Route:     ", solution["route"])
