# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC connection handling for the cuOpt MCP server.

``cuopt`` is imported lazily inside the accessors rather than at module
scope: importing it pulls the compiled ``libcuopt`` extension, and under
stdio that cost would be paid on every session start, delaying the
``initialize`` / ``tools/list`` handshake.
"""

import os
import re
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuopt.grpc.linear_programming import Client

DEFAULT_HOST = "localhost"
# Matches cuopt_default_grpc_port (cpp/src/grpc/cuopt_default_grpc_port.h) --
# cuopt_grpc_server's own default, not the unrelated routing-client examples
# that use 50051.
DEFAULT_PORT = 5001

# Redacts absolute filesystem paths out of backend error text, since a
# server-internal error (e.g. "Failed to open log file: <path>") can't be
# told apart from a caller-facing one by type alone.
_PATH_RE = re.compile(r"(?<!\w)(/[\w.\-]+){2,}")


def redact_paths(text: str) -> str:
    """Redact absolute filesystem paths from text bound for an MCP caller."""
    return _PATH_RE.sub("[redacted path]", text)


_lock = threading.Lock()
_client = None


def endpoint() -> tuple:
    """Return the configured ``(host, port)``.

    Reuses the same environment the cuOpt gRPC client already honours, so
    the MCP server introduces no new configuration surface.

    Raises
    ------
        CuOptMCPError: ``CUOPT_REMOTE_PORT`` isn't an integer, or is outside
            the valid TCP port range -- reported here so it's an actionable
            configuration error instead of a delayed connection failure.
    """
    host = os.environ.get("CUOPT_REMOTE_HOST", DEFAULT_HOST)
    raw_port = os.environ.get("CUOPT_REMOTE_PORT")
    if raw_port is None:
        return host, DEFAULT_PORT
    try:
        port = int(raw_port)
    except ValueError:
        raise CuOptMCPError(
            f"CUOPT_REMOTE_PORT={raw_port!r} is not an integer"
        ) from None
    if not 1 <= port <= 65535:
        raise CuOptMCPError(
            f"CUOPT_REMOTE_PORT={port} must be between 1 and 65535"
        )
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


def get_client() -> "Client":
    """Return a process-wide gRPC client, connecting on first use.

    The channel is the only state this process holds; jobs themselves live
    in cuopt_grpc_server and are addressed by the ``job_id`` returned to the
    caller, so a restart loses nothing but the connection.

    Returns
    -------
        The cached ``cuopt.grpc.linear_programming.Client``, creating it
        against the current ``CUOPT_REMOTE_HOST``/``CUOPT_REMOTE_PORT`` /
        TLS environment on first call.
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
    """Convert a backend exception into a model-facing :class:`CuOptMCPError`.

    Recognizes the unreachable-server case and names the host/port plus how
    to fix it. Any other backend text is passed through :func:`redact_paths`:
    the server's own error strings sometimes embed its internal paths (e.g.
    a job's log file location), which an MCP caller has no use for.

    Args:
        exc: The exception raised by the gRPC client call.

    Returns
    -------
        A :class:`CuOptMCPError` whose message is safe to return to the
        MCP caller.
    """
    host, port = endpoint()
    text = str(exc)
    if "UNAVAILABLE" in text or "failed to connect" in text.lower():
        return CuOptMCPError(
            f"cuOpt gRPC server unreachable at {host}:{port}. Start it with "
            f"`cuopt_grpc_server --port {port}`, or set CUOPT_REMOTE_HOST / "
            "CUOPT_REMOTE_PORT to point at a running server."
        )
    return CuOptMCPError(redact_paths(text))
