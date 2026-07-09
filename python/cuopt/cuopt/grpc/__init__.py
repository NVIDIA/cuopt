# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC clients for remote cuOpt execution.

This package is the namespace for domain-specific async clients:

- :mod:`cuopt.grpc.linear_programming` — LP/MILP/QP (submit, result, incumbents)
- :mod:`cuopt.grpc.routing` — VRP/TSP/PDP (future)

Shared job lifecycle (connect, status, wait, cancel, delete, logs) may live
here later as a base client type. Import domain clients explicitly, e.g.
``from cuopt.grpc.linear_programming import Client``.

Do not re-export ``Client`` from this package — callers must choose the
domain-specific client (``linear_programming``, ``routing``, etc.).
"""
