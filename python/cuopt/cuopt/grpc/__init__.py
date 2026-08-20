# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC clients for remote cuOpt execution.

This package is the namespace for domain-specific clients:

- :mod:`cuopt.grpc.linear_programming` — LP/MILP/QP (submit, result, incumbents)
- :mod:`cuopt.grpc.routing` — VRP/TSP/PDP

Both are re-exports from :mod:`cuopt.grpc.client`, a single extension module.
They wrap the same C++ client object, so they are compiled as one unit; the
split into two public modules is an API boundary, not a packaging one. Keeping
one unit is what allows the client to be detached from the solver engines as a
single GPU-free package later — two units would each need their own copy of the
transport layer.

Import domain clients explicitly, e.g.
``from cuopt.grpc.linear_programming import Client``.

Do not re-export ``Client`` from this package — callers must choose the
domain-specific client (``linear_programming``, ``routing``, etc.).
"""
