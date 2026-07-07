# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared cuopt_grpc_server helpers and pytest fixtures for LP tests.

Registered via ``pytest_plugins`` in ``python/cuopt/cuopt/tests/conftest.py``.

Class-scoped ``grpc_server`` starts one server per test class. Configure it on
the test class::

    class TestMyGrpcFeature:
        grpc_port_offset = GRPC_PORT_OFFSET_CLIENT
        grpc_server_yield = "port"   # or "env" (default)

        def test_foo(self, grpc_server):
            ...

Port offsets are added to ``CUOPT_TEST_PORT_BASE`` (default 18000) so parallel
test classes do not collide.
"""

import os
import shutil
import signal
import socket
import subprocess
import time

import pytest

# Port offsets (added to CUOPT_TEST_PORT_BASE). Keep unique per test class.
GRPC_PORT_OFFSET_CPU_ONLY = 600
GRPC_PORT_OFFSET_CLI = 700
GRPC_PORT_OFFSET_CLIENT = 800
GRPC_PORT_OFFSET_TLS = 850
GRPC_PORT_OFFSET_MTLS = 900


def find_grpc_server():
    """Locate cuopt_grpc_server binary."""
    env_path = os.environ.get("CUOPT_GRPC_SERVER_PATH")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    found = shutil.which("cuopt_grpc_server")
    if found:
        return found

    for candidate in [
        "./cuopt_grpc_server",
        "../cpp/build/cuopt_grpc_server",
        "../../cpp/build/cuopt_grpc_server",
    ]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        p = os.path.join(conda_prefix, "bin", "cuopt_grpc_server")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def wait_for_port(port, timeout=15):
    """Block until TCP port accepts connections or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def wait_for_grpc_client(port, timeout=30):
    """Block until cuopt.grpc.linear_programming.Client can connect (TCP up is not enough)."""
    from cuopt.grpc.linear_programming import Client, GrpcError

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not wait_for_port(port, timeout=1):
            time.sleep(0.2)
            continue
        try:
            try:
                client = Client("localhost", port, tls=False)
            except TypeError:
                client = Client("localhost", port)
            del client
            return True
        except GrpcError:
            time.sleep(0.2)
    return False


def client_remote_env(port):
    """Env for a CPU-only client process talking to a remote gRPC server."""
    env = os.environ.copy()
    for key in [k for k in env if k.startswith("CUOPT_TLS_")]:
        env.pop(key)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CUOPT_REMOTE_HOST"] = "localhost"
    env["CUOPT_REMOTE_PORT"] = str(port)
    return env


def server_env():
    """Env for ``cuopt_grpc_server`` — keep GPU access; drop client-only vars."""
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("CUOPT_TLS_") or key.startswith("CUOPT_REMOTE_"):
            env.pop(key)
    return env


# Backward-compatible alias used by tests that yield client env dicts.
cpu_only_env = client_remote_env


def start_grpc_server(port_offset):
    """Locate the server, start it on BASE + port_offset, return (proc, client_env)."""
    server_bin = find_grpc_server()
    if server_bin is None:
        pytest.skip("cuopt_grpc_server not found")

    port = int(os.environ.get("CUOPT_TEST_PORT_BASE", "18000")) + port_offset
    client_env = client_remote_env(port)
    proc = subprocess.Popen(
        [
            server_bin,
            "--port",
            str(port),
            "--workers",
            "1",
            "--log-to-console",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=server_env(),
    )
    time.sleep(0.5)
    if proc.poll() is not None:
        pytest.skip(
            f"cuopt_grpc_server exited immediately (rc={proc.returncode}), "
            "binary may be unable to load shared libraries in this environment"
        )
    if not wait_for_grpc_client(port, timeout=30):
        proc.kill()
        proc.wait()
        pytest.fail(
            "cuopt_grpc_server TCP port opened but gRPC client could not connect "
            "within 30s"
        )

    return proc, client_env


def stop_grpc_server(proc):
    """Gracefully shut down a server process."""
    if proc.poll() is not None:
        proc.wait()
        return

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="class")
def grpc_server(request):
    """Class-scoped server; see module docstring for configuration."""
    cls = request.cls
    if cls is None:
        pytest.fail("grpc_server requires a class-scoped test class")

    port_offset = getattr(cls, "grpc_port_offset", None)
    if port_offset is None:
        pytest.fail(
            f"{cls.__name__} must set grpc_port_offset "
            f"(e.g. GRPC_PORT_OFFSET_CLIENT)"
        )

    yield_kind = getattr(cls, "grpc_server_yield", "env")
    if yield_kind not in ("env", "port"):
        pytest.fail(
            f"{cls.__name__}.grpc_server_yield must be 'env' or 'port', "
            f"got {yield_kind!r}"
        )

    proc, client_env = start_grpc_server(port_offset)
    try:
        if yield_kind == "port":
            yield int(client_env["CUOPT_REMOTE_PORT"])
        else:
            yield client_env
    finally:
        stop_grpc_server(proc)
