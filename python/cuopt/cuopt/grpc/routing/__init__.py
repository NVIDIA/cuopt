# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled gRPC client for remote VRP (vehicle routing) solves.

Build a :class:`cuopt.routing.DataModel` and solve it on a remote
``cuopt_grpc_server``::

    from cuopt import routing
    from cuopt.grpc.routing import RoutingClient

    dm = routing.DataModel(n_locations, n_fleet)
    dm.add_cost_matrix(cost)
    ...
    client = RoutingClient("gpu-host:50051")
    solution = client.solve(dm)
"""

from cuopt.grpc.routing.grpc_client import RoutingClient, RoutingSolveError

__all__ = ["RoutingClient", "RoutingSolveError"]
