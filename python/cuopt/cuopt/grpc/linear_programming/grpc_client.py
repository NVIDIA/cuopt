# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim.

The LP/MIP and routing clients were merged into a single extension module
(:mod:`cuopt.grpc.client.grpc_client`). This module keeps the original import
path working.
"""

from cuopt.grpc.client.grpc_client import (  # noqa: F401
    Client,
    GrpcError,
    JobNotReadyError,
    JobStatus,
    TlsConfig,
)

__all__ = ["Client", "GrpcError", "JobNotReadyError", "JobStatus", "TlsConfig"]
