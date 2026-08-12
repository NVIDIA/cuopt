# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC connection handling for the cuOpt MCP server.

``cuopt`` is imported lazily inside the accessors rather than at module
scope: importing it pulls the compiled ``libcuopt`` extension, and under
stdio that cost would be paid on every session start, delaying the
``initialize`` / ``tools/list`` handshake.
"""

import os
import threading

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 50051

_lock = threading.Lock()
_client = None


def endpoint() -> tuple:
    """Return the configured ``(host, port)``.

    Reuses the same environment the cuOpt gRPC client already honours, so
    the MCP server introduces no new configuration surface.
    """
    host = os.environ.get("CUOPT_REMOTE_HOST", DEFAULT_HOST)
    port = int(os.environ.get("CUOPT_REMOTE_PORT", DEFAULT_PORT))
    return host, port


def _tls_config():
    if os.environ.get("CUOPT_TLS_ENABLED", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return None
    from cuopt.grpc.linear_programming import TlsConfig

    return TlsConfig(
        root_certs=os.environ.get("CUOPT_TLS_ROOT_CERT"),
        client_cert=os.environ.get("CUOPT_TLS_CLIENT_CERT"),
        client_key=os.environ.get("CUOPT_TLS_CLIENT_KEY"),
    )


def get_client():
    """Return a process-wide gRPC client, connecting on first use.

    The channel is the only state this process holds; jobs themselves live
    in cuopt_grpc_server and are addressed by the ``job_id`` returned to the
    caller, so a restart loses nothing but the connection.
    """
    global _client
    with _lock:
        if _client is None:
            from cuopt.grpc.linear_programming import Client

            host, port = endpoint()
            _client = Client(host, port, tls=_tls_config())
        return _client


def reset_client() -> None:
    """Drop the cached client. Used by tests and after a fatal channel error."""
    global _client
    with _lock:
        _client = None


class CuOptMCPError(RuntimeError):
    """Raised with text meant for the model, not a stack trace."""


def describe_connection_error(exc: Exception) -> CuOptMCPError:
    host, port = endpoint()
    text = str(exc)
    if "UNAVAILABLE" in text or "failed to connect" in text.lower():
        return CuOptMCPError(
            f"cuOpt gRPC server unreachable at {host}:{port}. Start it with "
            f"`cuopt_grpc_server --port {port}`, or set CUOPT_REMOTE_HOST / "
            "CUOPT_REMOTE_PORT to point at a running server."
        )
    return CuOptMCPError(text)
