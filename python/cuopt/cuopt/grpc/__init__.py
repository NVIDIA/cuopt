# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC clients for remote cuOpt execution.

This package is the namespace for domain-specific async clients:

- :mod:`cuopt.grpc.mathematical_optimization` — LP/MILP/QP (submit, result, incumbents)
- :mod:`cuopt.grpc.routing` — VRP/TSP/PDP (future)

Longer term, shared job lifecycle (connect, status, wait, cancel, delete,
logs) may live here as a base client type that domain clients extend or
compose. Import domain clients explicitly, e.g.
``from cuopt.grpc.mathematical_optimization import Client``.
"""
