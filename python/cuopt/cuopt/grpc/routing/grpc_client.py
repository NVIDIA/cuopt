# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim.

The routing and LP/MIP clients were merged into a single extension module
(:mod:`cuopt.grpc.client.grpc_client`). This module keeps the original import
path working, including the names the serialization test reaches for
(``HANDLED_SETTERS``, ``problem_summary``).
"""

from cuopt.grpc.client.grpc_client import (  # noqa: F401
    HANDLED_SETTERS,
    RoutingClient,
    RoutingSolveError,
    problem_summary,
)

__all__ = [
    "HANDLED_SETTERS",
    "RoutingClient",
    "RoutingSolveError",
    "problem_summary",
]
