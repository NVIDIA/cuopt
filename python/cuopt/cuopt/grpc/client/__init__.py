# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled gRPC clients for remote cuOpt solves (LP/MIP and routing).

Prefer the domain-specific entry points, which re-export from here:
:mod:`cuopt.grpc.linear_programming` and :mod:`cuopt.grpc.routing`.
"""

from cuopt.grpc.client.grpc_client import (
    Client,
    GrpcError,
    HANDLED_SETTERS,
    JobNotReadyError,
    JobStatus,
    RoutingClient,
    RoutingSolveError,
    TlsConfig,
    problem_summary,
)

__all__ = [
    "Client",
    "GrpcError",
    "HANDLED_SETTERS",
    "JobNotReadyError",
    "JobStatus",
    "RoutingClient",
    "RoutingSolveError",
    "TlsConfig",
    "problem_summary",
]
