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

import logging
import os
import shutil
import signal
import socket
import subprocess
import time

import pytest

logger = logging.getLogger(__name__)

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


def _set_pdeathsig():
    """SIGKILL this child if the spawning worker dies (Linux; best effort)."""
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            PR_SET_PDEATHSIG, signal.SIGKILL
        )
    except Exception:
        pass


def spawn_server(cmd, env=None):
    """Start ``cuopt_grpc_server`` in its own process group, dying with the
    spawning worker, so it and its ``--workers`` child can be reaped together
    (see ``kill_server``) and don't leak GPU memory.
    """
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
        preexec_fn=_set_pdeathsig,
    )


def kill_server(proc):
    """Terminate the server's whole process group (parent + ``--workers`` child)."""
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


# Backward-compatible alias used by tests that yield client env dicts.
cpu_only_env = client_remote_env


def generate_test_certs(cert_dir):
    """Generate a CA, server cert, and client cert for TLS/mTLS tests."""
    if not shutil.which("openssl"):
        return False

    def _run(cmd):
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            logger.warning(
                "cert command failed: %s (rc=%d)\nstdout: %s\nstderr: %s",
                cmd,
                result.returncode,
                result.stdout.decode(errors="replace"),
                result.stderr.decode(errors="replace"),
            )
            return False
        return True

    ca_key = os.path.join(cert_dir, "ca.key")
    ca_crt = os.path.join(cert_dir, "ca.crt")
    if not _run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            ca_key,
            "-out",
            ca_crt,
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=TestCA",
        ]
    ):
        return False

    server_key = os.path.join(cert_dir, "server.key")
    server_csr = os.path.join(cert_dir, "server.csr")
    server_crt = os.path.join(cert_dir, "server.crt")
    server_ext = os.path.join(cert_dir, "server.ext")
    if not _run(
        [
            "openssl",
            "req",
            "-newkey",
            "rsa:2048",
            "-keyout",
            server_key,
            "-out",
            server_csr,
            "-nodes",
            "-subj",
            "/CN=localhost",
        ]
    ):
        return False
    with open(server_ext, "w", encoding="utf-8") as f:
        f.write("subjectAltName=DNS:localhost,IP:127.0.0.1\n")
    if not _run(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            server_csr,
            "-CA",
            ca_crt,
            "-CAkey",
            ca_key,
            "-CAcreateserial",
            "-out",
            server_crt,
            "-days",
            "1",
            "-extfile",
            server_ext,
        ]
    ):
        return False

    client_key = os.path.join(cert_dir, "client.key")
    client_csr = os.path.join(cert_dir, "client.csr")
    client_crt = os.path.join(cert_dir, "client.crt")
    if not _run(
        [
            "openssl",
            "req",
            "-newkey",
            "rsa:2048",
            "-keyout",
            client_key,
            "-out",
            client_csr,
            "-nodes",
            "-subj",
            "/CN=TestClient",
        ]
    ):
        return False
    if not _run(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            client_csr,
            "-CA",
            ca_crt,
            "-CAkey",
            ca_key,
            "-CAcreateserial",
            "-out",
            client_crt,
            "-days",
            "1",
        ]
    ):
        return False

    return True


def start_tls_grpc_server(port_offset, cert_dir, require_client_cert=False):
    """Start a TLS-enabled cuopt_grpc_server and return (proc, port)."""
    server_bin = find_grpc_server()
    if server_bin is None:
        pytest.skip("cuopt_grpc_server not found")

    port = int(os.environ.get("CUOPT_TEST_PORT_BASE", "18000")) + port_offset
    args = [
        server_bin,
        "--port",
        str(port),
        "--workers",
        "1",
        "--tls",
        "--tls-cert",
        os.path.join(cert_dir, "server.crt"),
        "--tls-key",
        os.path.join(cert_dir, "server.key"),
    ]
    if require_client_cert:
        args.extend(
            [
                "--tls-root",
                os.path.join(cert_dir, "ca.crt"),
                "--require-client-cert",
            ]
        )

    proc = spawn_server(
        args,
        env=server_env(),
    )
    time.sleep(0.5)
    if proc.poll() is not None:
        pytest.skip(
            f"cuopt_grpc_server exited immediately (rc={proc.returncode}), "
            "binary may be unable to load shared libraries in this environment"
        )
    if not wait_for_port(port, timeout=15):
        kill_server(proc)
        pytest.fail("TLS cuopt_grpc_server failed to start within 15s")

    return proc, port


def start_grpc_server(port_offset):
    """Locate the server, start it on BASE + port_offset, return (proc, client_env)."""
    server_bin = find_grpc_server()
    if server_bin is None:
        pytest.skip("cuopt_grpc_server not found")

    port = int(os.environ.get("CUOPT_TEST_PORT_BASE", "18000")) + port_offset
    client_env = client_remote_env(port)
    proc = spawn_server(
        [
            server_bin,
            "--port",
            str(port),
            "--workers",
            "1",
            "--log-to-console",
        ],
        env=server_env(),
    )
    time.sleep(0.5)
    if proc.poll() is not None:
        pytest.skip(
            f"cuopt_grpc_server exited immediately (rc={proc.returncode}), "
            "binary may be unable to load shared libraries in this environment"
        )
    if not wait_for_grpc_client(port, timeout=30):
        kill_server(proc)
        pytest.fail(
            "cuopt_grpc_server TCP port opened but gRPC client could not connect "
            "within 30s"
        )

    return proc, client_env


def stop_grpc_server(proc):
    """Gracefully shut down a server process and its worker child."""
    kill_server(proc)


def client_tls_env(port, cert_dir, mtls=False):
    """Return an env dict for remote execution over TLS (or mTLS)."""
    env = client_remote_env(port)
    env["CUOPT_TLS_ENABLED"] = "1"
    env["CUOPT_TLS_ROOT_CERT"] = os.path.join(cert_dir, "ca.crt")
    if mtls:
        env["CUOPT_TLS_CLIENT_CERT"] = os.path.join(cert_dir, "client.crt")
        env["CUOPT_TLS_CLIENT_KEY"] = os.path.join(cert_dir, "client.key")
    return env


def resolve_test_port(port_offset):
    """Return BASE + port_offset, plus xdist worker id when running under xdist."""
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    worker_id = int(worker[2:]) if worker.startswith("gw") else 0
    return (
        int(os.environ.get("CUOPT_TEST_PORT_BASE", "18000"))
        + port_offset
        + worker_id
    )


def start_subprocess_grpc_server(port_offset):
    """Start plaintext cuopt_grpc_server for subprocess remote-execution tests."""
    server_bin = find_grpc_server()
    if server_bin is None:
        pytest.skip("cuopt_grpc_server not found")

    port = resolve_test_port(port_offset)
    proc = spawn_server(
        [server_bin, "--port", str(port), "--workers", "1"],
    )
    time.sleep(0.5)
    if proc.poll() is not None:
        pytest.skip(
            f"cuopt_grpc_server exited immediately (rc={proc.returncode}), "
            "binary may be unable to load shared libraries in this environment"
        )
    if not wait_for_port(port, timeout=15):
        kill_server(proc)
        pytest.fail("cuopt_grpc_server failed to start within 15s")

    return proc, client_remote_env(port)


@pytest.fixture(scope="class")
def tls_server_info(tmp_path_factory):
    """TLS server plus cert directory for in-process ``Client(tls=...)`` tests."""
    cert_dir = str(tmp_path_factory.mktemp("grpc_tls_certs"))
    if not generate_test_certs(cert_dir):
        pytest.skip("openssl not available or cert generation failed")

    proc, port = start_tls_grpc_server(GRPC_PORT_OFFSET_TLS, cert_dir)
    try:
        yield {"port": port, "cert_dir": cert_dir}
    finally:
        stop_grpc_server(proc)


@pytest.fixture(scope="class")
def mtls_server_info(tmp_path_factory):
    """Mutual-TLS server (client cert required) plus cert directory."""
    cert_dir = str(tmp_path_factory.mktemp("grpc_mtls_certs"))
    if not generate_test_certs(cert_dir):
        pytest.skip("openssl not available or cert generation failed")

    proc, port = start_tls_grpc_server(
        GRPC_PORT_OFFSET_MTLS, cert_dir, require_client_cert=True
    )
    try:
        yield {"port": port, "cert_dir": cert_dir}
    finally:
        stop_grpc_server(proc)


@pytest.fixture(scope="class")
def tls_env_with_server(tmp_path_factory):
    """TLS server env dict for subprocess ``Problem.solve()`` remote tests."""
    cert_dir = str(tmp_path_factory.mktemp("tls_certs"))
    if not generate_test_certs(cert_dir):
        pytest.skip("openssl not available or cert generation failed")

    proc, port = start_tls_grpc_server(GRPC_PORT_OFFSET_TLS, cert_dir)
    try:
        yield client_tls_env(port, cert_dir, mtls=False)
    finally:
        stop_grpc_server(proc)


@pytest.fixture(scope="class")
def mtls_env_with_server(mtls_server_info):
    """Mutual-TLS client env dict for subprocess remote-execution tests."""
    yield client_tls_env(
        mtls_server_info["port"],
        mtls_server_info["cert_dir"],
        mtls=True,
    )


@pytest.fixture(scope="class")
def cpu_only_env_with_server():
    """Plaintext server env for CPU-only subprocess tests."""
    proc, env = start_subprocess_grpc_server(GRPC_PORT_OFFSET_CPU_ONLY)
    try:
        yield env
    finally:
        stop_grpc_server(proc)


@pytest.fixture(scope="class")
def cli_remote_env_with_server():
    """Plaintext server env for cuopt_cli subprocess remote tests."""
    proc, env = start_subprocess_grpc_server(GRPC_PORT_OFFSET_CLI)
    try:
        yield env
    finally:
        stop_grpc_server(proc)


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
