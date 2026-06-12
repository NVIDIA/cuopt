# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared cuopt_grpc_server helpers and pytest fixtures for LP tests.

Import this module in test files that need a live server::

    pytest_plugins = ["grpc_server_fixtures"]

Class-scoped fixtures start one server per test class and tear it down after
the class finishes. Port offsets are added to ``CUOPT_TEST_PORT_BASE`` (default
18000) so parallel test classes do not collide.
"""

import os
import shutil
import signal
import socket
import subprocess
import time

import pytest

# Port offsets (added to CUOPT_TEST_PORT_BASE). Keep unique per fixture/class.
GRPC_PORT_OFFSET_CPU_ONLY = 600
GRPC_PORT_OFFSET_CLI = 700
GRPC_PORT_OFFSET_CLIENT = 800
GRPC_PORT_OFFSET_TLS = 800
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


def cpu_only_env(port):
    """Return an env dict that hides all GPUs and enables remote mode."""
    env = os.environ.copy()
    for key in [k for k in env if k.startswith("CUOPT_TLS_")]:
        env.pop(key)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CUOPT_REMOTE_HOST"] = "localhost"
    env["CUOPT_REMOTE_PORT"] = str(port)
    return env


def start_grpc_server(port_offset):
    """Locate the server, start it on BASE + port_offset, return (proc, env)."""
    server_bin = find_grpc_server()
    if server_bin is None:
        pytest.skip("cuopt_grpc_server not found")

    port = int(os.environ.get("CUOPT_TEST_PORT_BASE", "18000")) + port_offset
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
    )
    time.sleep(0.5)
    if proc.poll() is not None:
        pytest.skip(
            f"cuopt_grpc_server exited immediately (rc={proc.returncode}), "
            "binary may be unable to load shared libraries in this environment"
        )
    if not wait_for_port(port, timeout=15):
        proc.kill()
        proc.wait()
        pytest.fail("cuopt_grpc_server failed to start within 15s")

    return proc, cpu_only_env(port)


def stop_grpc_server(proc):
    """Gracefully shut down a server process."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="class")
def grpc_cpu_only_env():
    """CPU-only remote env (``CUOPT_REMOTE_*``) with server on offset 600."""
    proc, env = start_grpc_server(GRPC_PORT_OFFSET_CPU_ONLY)
    yield env
    stop_grpc_server(proc)


@pytest.fixture(scope="class")
def grpc_cli_cpu_only_env():
    """CPU-only remote env with server on offset 700 (cuopt_cli tests)."""
    proc, env = start_grpc_server(GRPC_PORT_OFFSET_CLI)
    yield env
    stop_grpc_server(proc)


@pytest.fixture(scope="class")
def grpc_client_port():
    """Listening port for ``cuopt.grpc.Client`` tests (offset 800)."""
    proc, env = start_grpc_server(GRPC_PORT_OFFSET_CLIENT)
    yield int(env["CUOPT_REMOTE_PORT"])
    stop_grpc_server(proc)
