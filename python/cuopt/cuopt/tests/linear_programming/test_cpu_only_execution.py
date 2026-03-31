# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for CPU-only execution mode and solution interface polymorphism.

These tests verify that cuOpt can run on a CPU host without GPU access,
forwarding solves to a real cuopt_grpc_server over gRPC. A single shared
server is started once per test class to avoid per-test startup overhead.

TestSolutionInterfacePolymorphism:
    Run in-process on real GPU hardware and assert correctness of
    solution values against known optima.
"""

import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

import cuopt_mps_parser
import pytest
from cuopt import linear_programming
from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT

logger = logging.getLogger(__name__)

RAPIDS_DATASET_ROOT_DIR = os.environ.get(
    "RAPIDS_DATASET_ROOT_DIR", "./datasets"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_grpc_server():
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


def _wait_for_port(port, timeout=15):
    """Block until TCP port accepts connections or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _cpu_only_env(port):
    """Return an env dict that hides all GPUs and enables remote mode."""
    env = os.environ.copy()
    for key in [k for k in env if k.startswith("CUOPT_TLS_")]:
        env.pop(key)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CUOPT_REMOTE_HOST"] = "localhost"
    env["CUOPT_REMOTE_PORT"] = str(port)
    return env


def _parse_cli_output(output):
    """Extract solver status and objective value from cuopt_cli output.

    Handles both the LP summary format
        (``Status: Optimal  Objective: -464.753  ...  Time: 0.1s``)
    and the MIP format
        (``Optimal solution found.`` + ``Solution objective: 2.000000 ...``).
    """
    result = {"status": "Unknown", "objective_value": float("nan")}

    for line in output.split("\n"):
        stripped = line.strip()

        # LP summary: "Status: Optimal  Objective: -464.753  ... Time: 0.1s"
        if stripped.startswith("Status:") and "Time:" in stripped:
            m = re.match(r"Status:\s*(\S+)", stripped)
            if m:
                result["status"] = m.group(1)
            m = re.search(
                r"Objective:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
                stripped,
            )
            if m:
                result["objective_value"] = float(m.group(1))
            continue

        # MIP termination
        if stripped == "Optimal solution found.":
            result["status"] = "Optimal"
            continue

        # MIP solution: "Solution objective: 2.000000 , ..."
        m = re.match(
            r"Solution objective:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
            stripped,
        )
        if m:
            result["objective_value"] = float(m.group(1))
            continue

    return result


def _generate_test_certs(cert_dir):
    """Generate a CA, server cert, and client cert for TLS/mTLS tests.

    Returns True on success, False if openssl is missing or a command fails.
    """
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
    with open(server_ext, "w") as f:
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


def _tls_env(port, cert_dir, mtls=False):
    """Return an env dict for remote execution over TLS (or mTLS)."""
    env = _cpu_only_env(port)
    env["CUOPT_TLS_ENABLED"] = "1"
    env["CUOPT_TLS_ROOT_CERT"] = os.path.join(cert_dir, "ca.crt")
    if mtls:
        env["CUOPT_TLS_CLIENT_CERT"] = os.path.join(cert_dir, "client.crt")
        env["CUOPT_TLS_CLIENT_KEY"] = os.path.join(cert_dir, "client.key")
    return env


def _run_in_subprocess(func, env=None, timeout=120):
    """Run *func* (a top-level function in this module) in a fresh subprocess."""
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), func.__name__],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


# ---------------------------------------------------------------------------
# CPU-only subprocess test implementations
#
# Each function below runs in an isolated subprocess with
# CUDA_VISIBLE_DEVICES="" so the CUDA driver is never initialized.
# Imports are local so the subprocess doesn't need GPU-dependent modules
# at the top level.
# ---------------------------------------------------------------------------


def _impl_lp_solve_cpu_only():
    """LP solve returns correctly-sized solution vectors."""
    from cuopt import linear_programming
    import cuopt_mps_parser

    dataset_root = os.environ.get("RAPIDS_DATASET_ROOT_DIR", "./")
    mps_file = f"{dataset_root}/linear_programming/afiro_original.mps"
    dm = cuopt_mps_parser.ParseMps(mps_file)
    n_vars = len(dm.get_objective_coefficients())

    solution = linear_programming.Solve(
        dm, linear_programming.SolverSettings()
    )

    primal = solution.get_primal_solution()
    assert len(primal) == n_vars, (
        f"primal size {len(primal)} != n_vars {n_vars}"
    )

    obj = solution.get_primal_objective()
    assert obj is not None, "objective is None"

    _AFIRO_OBJ = -464.7531428571
    rel_err = abs(obj - _AFIRO_OBJ) / max(abs(_AFIRO_OBJ), 1e-12)
    assert rel_err < 0.01, (
        f"objective {obj} differs from expected {_AFIRO_OBJ} "
        f"(rel error {rel_err:.4e})"
    )


def _impl_lp_dual_solution_cpu_only():
    """Dual solution and reduced costs are correctly sized."""
    from cuopt import linear_programming
    import cuopt_mps_parser

    dataset_root = os.environ.get("RAPIDS_DATASET_ROOT_DIR", "./")
    mps_file = f"{dataset_root}/linear_programming/afiro_original.mps"
    dm = cuopt_mps_parser.ParseMps(mps_file)
    n_vars = len(dm.get_objective_coefficients())
    n_cons = len(dm.get_constraint_bounds())

    solution = linear_programming.Solve(
        dm, linear_programming.SolverSettings()
    )

    dual = solution.get_dual_solution()
    assert len(dual) == n_cons, f"dual size {len(dual)} != n_cons {n_cons}"

    rc = solution.get_reduced_cost()
    assert len(rc) == n_vars, f"reduced_cost size {len(rc)} != n_vars {n_vars}"

    obj = solution.get_primal_objective()
    _AFIRO_OBJ = -464.7531428571
    rel_err = abs(obj - _AFIRO_OBJ) / max(abs(_AFIRO_OBJ), 1e-12)
    assert rel_err < 0.01, (
        f"dual test: objective {obj} differs from expected {_AFIRO_OBJ} "
        f"(rel error {rel_err:.4e})"
    )


def _impl_mip_solve_cpu_only():
    """MIP solve returns correctly-sized solution vector."""
    from cuopt import linear_programming
    from cuopt.linear_programming.solver.solver_parameters import (
        CUOPT_TIME_LIMIT,
    )
    import cuopt_mps_parser

    dataset_root = os.environ.get("RAPIDS_DATASET_ROOT_DIR", "./")
    mps_file = f"{dataset_root}/mip/bb_optimality.mps"
    dm = cuopt_mps_parser.ParseMps(mps_file)
    n_vars = len(dm.get_objective_coefficients())

    settings = linear_programming.SolverSettings()
    settings.set_parameter(CUOPT_TIME_LIMIT, 60.0)

    solution = linear_programming.Solve(dm, settings)
    vals = solution.get_primal_solution()
    assert len(vals) == n_vars, f"solution size {len(vals)} != n_vars {n_vars}"

    obj_coeffs = dm.get_objective_coefficients()
    computed_obj = sum(c * v for c, v in zip(obj_coeffs, vals))
    reported_obj = solution.get_primal_objective()
    if abs(reported_obj) > 1e-12:
        rel_err = abs(computed_obj - reported_obj) / abs(reported_obj)
    else:
        rel_err = abs(computed_obj - reported_obj)
    assert rel_err < 0.01, (
        f"MIP objective mismatch: computed {computed_obj} vs reported "
        f"{reported_obj} (rel error {rel_err:.4e})"
    )


def _impl_warmstart_cpu_only():
    """Warmstart round-trip works without touching CUDA."""
    from cuopt import linear_programming
    from cuopt.linear_programming.solver.solver_parameters import (
        CUOPT_METHOD,
        CUOPT_ITERATION_LIMIT,
        CUOPT_PRESOLVE,
    )
    from cuopt.linear_programming.solver_settings import SolverMethod
    import cuopt_mps_parser

    dataset_root = os.environ.get("RAPIDS_DATASET_ROOT_DIR", "./")
    mps_file = f"{dataset_root}/linear_programming/afiro_original.mps"
    dm = cuopt_mps_parser.ParseMps(mps_file)

    settings = linear_programming.SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
    settings.set_parameter(CUOPT_PRESOLVE, 0)
    settings.set_parameter(CUOPT_ITERATION_LIMIT, 100)

    sol1 = linear_programming.Solve(dm, settings)
    ws = sol1.get_pdlp_warm_start_data()

    if ws is not None:
        settings.set_pdlp_warm_start_data(ws)
        settings.set_parameter(CUOPT_ITERATION_LIMIT, 200)
        sol2 = linear_programming.Solve(dm, settings)
        assert sol2.get_primal_solution() is not None


# ---------------------------------------------------------------------------
# Shared fixture helpers (used by TestCPUOnlyExecution and TestCuoptCliCPUOnly)
# ---------------------------------------------------------------------------


def _start_grpc_server_fixture(port_offset):
    """Locate the server, start it on BASE + port_offset, return (proc, env)."""
    server_bin = _find_grpc_server()
    if server_bin is None:
        pytest.skip("cuopt_grpc_server not found")

    port = int(os.environ.get("CUOPT_TEST_PORT_BASE", "18000")) + port_offset
    proc = subprocess.Popen(
        [server_bin, "--port", str(port), "--workers", "1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _wait_for_port(port, timeout=15):
        proc.kill()
        proc.wait()
        pytest.fail("cuopt_grpc_server failed to start within 15s")

    return proc, _cpu_only_env(port)


def _stop_grpc_server(proc):
    """Gracefully shut down a server process."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# CPU-only Python tests  (subprocess required)
# ---------------------------------------------------------------------------


class TestCPUOnlyExecution:
    """Tests that run with CUDA_VISIBLE_DEVICES='' to simulate CPU-only hosts.

    A shared cuopt_grpc_server is started once for the whole class.
    """

    @pytest.fixture(scope="class")
    def cpu_only_env_with_server(self):
        proc, env = _start_grpc_server_fixture(port_offset=600)
        yield env
        _stop_grpc_server(proc)

    def test_lp_solve_cpu_only(self, cpu_only_env_with_server):
        """LP solve returns correctly-sized solution vectors."""
        result = _run_in_subprocess(
            _impl_lp_solve_cpu_only, env=cpu_only_env_with_server
        )
        assert result.returncode == 0, f"Test failed:\n{result.stderr}"

    def test_lp_dual_solution_cpu_only(self, cpu_only_env_with_server):
        """Dual solution and reduced costs are correctly sized."""
        result = _run_in_subprocess(
            _impl_lp_dual_solution_cpu_only, env=cpu_only_env_with_server
        )
        assert result.returncode == 0, f"Test failed:\n{result.stderr}"

    def test_mip_solve_cpu_only(self, cpu_only_env_with_server):
        """MIP solve returns correctly-sized solution vector."""
        result = _run_in_subprocess(
            _impl_mip_solve_cpu_only, env=cpu_only_env_with_server
        )
        assert result.returncode == 0, f"Test failed:\n{result.stderr}"

    def test_warmstart_cpu_only(self, cpu_only_env_with_server):
        """Warmstart round-trip works without touching CUDA."""
        result = _run_in_subprocess(
            _impl_warmstart_cpu_only, env=cpu_only_env_with_server
        )
        assert result.returncode == 0, f"Test failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# CPU-only CLI tests  (subprocess inherently needed)
# ---------------------------------------------------------------------------


class TestCuoptCliCPUOnly:
    """Test that cuopt_cli runs without CUDA in remote-execution mode.

    A shared cuopt_grpc_server is started once for the whole class.
    """

    @pytest.fixture(scope="class")
    def cpu_only_env_with_server(self):
        proc, env = _start_grpc_server_fixture(port_offset=700)
        yield env
        _stop_grpc_server(proc)

    @staticmethod
    def _find_cuopt_cli():
        for loc in [
            shutil.which("cuopt_cli"),
            "./cuopt_cli",
            "../cpp/build/cuopt_cli",
            "../../cpp/build/cuopt_cli",
        ]:
            if loc and os.path.isfile(loc) and os.access(loc, os.X_OK):
                return os.path.abspath(loc)

        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            p = os.path.join(conda_prefix, "bin", "cuopt_cli")
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
        return None

    _CUDA_ERRORS = [
        "CUDA error",
        "cudaErrorNoDevice",
        "no CUDA-capable device",
        "CUDA driver version is insufficient",
        "CUDA initialization failed",
    ]

    def _run_cli(self, mps_file, env, extra_args=None):
        """Run cuopt_cli on *mps_file* in remote-execution mode.

        Returns the combined stdout+stderr so callers can parse it.
        Asserts no CUDA errors and zero exit code.
        """
        cli = self._find_cuopt_cli()
        if cli is None:
            pytest.skip("cuopt_cli not found")
        if not os.path.exists(mps_file):
            pytest.skip(f"Test file not found: {mps_file}")

        cmd = [cli, mps_file, "--time-limit", "60"]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        print(result.stdout)
        print(result.stderr)

        combined = result.stdout + result.stderr
        for err in self._CUDA_ERRORS:
            assert err not in combined, (
                f"CUDA error '{err}' in output -- "
                "cuopt_cli must not require CUDA in remote mode"
            )
        assert result.returncode == 0, (
            f"cuopt_cli exited with {result.returncode}"
        )
        return combined

    _REMOTE_INDICATORS = [
        "Using remote GPU backend",
    ]

    def _assert_remote_execution(self, output):
        """Check that log output contains evidence of remote gRPC execution."""
        for indicator in self._REMOTE_INDICATORS:
            assert indicator in output, (
                f"Remote execution indicator '{indicator}' not found "
                "in CLI output -- solve may not have been forwarded"
            )

    def test_cli_lp_remote(self, cpu_only_env_with_server):
        """LP solve via cuopt_cli runs remotely with correct objective."""
        output = self._run_cli(
            f"{RAPIDS_DATASET_ROOT_DIR}/linear_programming/afiro_original.mps",
            cpu_only_env_with_server,
        )
        self._assert_remote_execution(output)

        parsed = _parse_cli_output(output)
        assert parsed["status"] == "Optimal", (
            f"Expected Optimal, got {parsed['status']}"
        )
        expected_obj = -464.7531428571
        rel_err = abs(parsed["objective_value"] - expected_obj) / abs(
            expected_obj
        )
        assert rel_err < 0.01, (
            f"Objective {parsed['objective_value']} differs from expected "
            f"{expected_obj} (rel error {rel_err:.4e})"
        )

    def test_cli_mip_remote(self, cpu_only_env_with_server):
        """MIP solve via cuopt_cli runs remotely with correct objective."""
        output = self._run_cli(
            f"{RAPIDS_DATASET_ROOT_DIR}/mip/bb_optimality.mps",
            cpu_only_env_with_server,
        )
        self._assert_remote_execution(output)

        parsed = _parse_cli_output(output)
        assert parsed["status"] == "Optimal", (
            f"Expected Optimal, got {parsed['status']}"
        )
        expected_obj = 2.0
        rel_err = abs(parsed["objective_value"] - expected_obj) / max(
            abs(expected_obj), 1e-12
        )
        assert rel_err < 0.01, (
            f"Objective {parsed['objective_value']} differs from expected "
            f"{expected_obj} (rel error {rel_err:.4e})"
        )


# ---------------------------------------------------------------------------
# Solution-interface tests  (in-process, real GPU, assert correctness)
# ---------------------------------------------------------------------------

_AFIRO_OBJ = -464.7531428571


class TestSolutionInterfacePolymorphism:
    """Verify solution accessors return correct values on real GPU solves."""

    def test_lp_solution_values(self):
        """LP solve of afiro.mps returns correct objective and sizes."""
        mps_file = (
            f"{RAPIDS_DATASET_ROOT_DIR}/linear_programming/afiro_original.mps"
        )
        dm = cuopt_mps_parser.ParseMps(mps_file)
        n_vars = len(dm.get_objective_coefficients())
        n_cons = len(dm.get_constraint_bounds())

        solution = linear_programming.Solve(
            dm, linear_programming.SolverSettings()
        )

        primal = solution.get_primal_solution()
        assert len(primal) == n_vars

        obj = solution.get_primal_objective()
        assert abs(obj - _AFIRO_OBJ) / abs(_AFIRO_OBJ) < 0.01, (
            f"Objective {obj} too far from expected {_AFIRO_OBJ}"
        )

        dual = solution.get_dual_solution()
        assert len(dual) == n_cons

        rc = solution.get_reduced_cost()
        assert len(rc) == n_vars

        dual_obj = solution.get_dual_objective()
        if obj != 0:
            assert abs(dual_obj - obj) / abs(obj) < 0.05

    def test_mip_solution_values(self):
        """MIP solve of bb_optimality.mps returns valid stats."""
        mps_file = f"{RAPIDS_DATASET_ROOT_DIR}/mip/bb_optimality.mps"
        dm = cuopt_mps_parser.ParseMps(mps_file)
        n_vars = len(dm.get_objective_coefficients())

        settings = linear_programming.SolverSettings()
        settings.set_parameter(CUOPT_TIME_LIMIT, 60.0)

        solution = linear_programming.Solve(dm, settings)

        vals = solution.get_primal_solution()
        assert len(vals) == n_vars

        stats = solution.get_milp_stats()
        assert "mip_gap" in stats
        assert "solution_bound" in stats
        assert stats["mip_gap"] >= 0, f"Negative MIP gap: {stats['mip_gap']}"


# ---------------------------------------------------------------------------
# TLS tests  (subprocess required, server with --tls)
# ---------------------------------------------------------------------------


class TestTLSExecution:
    """Test remote execution over a TLS-encrypted channel.

    A shared cuopt_grpc_server is started with --tls and self-signed certs.
    The client connects using CUOPT_TLS_* env vars.
    """

    @pytest.fixture(scope="class")
    def tls_env_with_server(self, tmp_path_factory):
        cert_dir = str(tmp_path_factory.mktemp("tls_certs"))
        if not _generate_test_certs(cert_dir):
            pytest.skip("openssl not available or cert generation failed")

        server_bin = _find_grpc_server()
        if server_bin is None:
            pytest.skip("cuopt_grpc_server not found")

        port = int(os.environ.get("CUOPT_TEST_PORT_BASE", "18000")) + 800
        proc = subprocess.Popen(
            [
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
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not _wait_for_port(port, timeout=15):
            proc.kill()
            proc.wait()
            pytest.fail("TLS cuopt_grpc_server failed to start within 15s")

        env = _tls_env(port, cert_dir, mtls=False)
        yield env

        _stop_grpc_server(proc)

    def test_lp_solve_tls(self, tls_env_with_server):
        """LP solve succeeds over a TLS channel."""
        result = _run_in_subprocess(
            _impl_lp_solve_cpu_only, env=tls_env_with_server
        )
        assert result.returncode == 0, f"TLS LP solve failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# mTLS tests  (subprocess required, server with --tls + --require-client-cert)
# ---------------------------------------------------------------------------


class TestMTLSExecution:
    """Test remote execution over an mTLS-encrypted channel.

    A shared cuopt_grpc_server is started with --tls, --tls-root, and
    --require-client-cert. The client must present a valid certificate
    signed by the test CA.
    """

    @pytest.fixture(scope="class")
    def mtls_server_info(self, tmp_path_factory):
        cert_dir = str(tmp_path_factory.mktemp("mtls_certs"))
        if not _generate_test_certs(cert_dir):
            pytest.skip("openssl not available or cert generation failed")

        server_bin = _find_grpc_server()
        if server_bin is None:
            pytest.skip("cuopt_grpc_server not found")

        port = int(os.environ.get("CUOPT_TEST_PORT_BASE", "18000")) + 900
        proc = subprocess.Popen(
            [
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
                "--tls-root",
                os.path.join(cert_dir, "ca.crt"),
                "--require-client-cert",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not _wait_for_port(port, timeout=15):
            proc.kill()
            proc.wait()
            pytest.fail("mTLS cuopt_grpc_server failed to start within 15s")

        yield {"port": port, "cert_dir": cert_dir}

        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def test_lp_solve_mtls(self, mtls_server_info):
        """LP solve succeeds over an mTLS channel with valid client cert."""
        env = _tls_env(
            mtls_server_info["port"],
            mtls_server_info["cert_dir"],
            mtls=True,
        )
        result = _run_in_subprocess(_impl_lp_solve_cpu_only, env=env)
        assert result.returncode == 0, (
            f"mTLS LP solve failed:\n{result.stderr}"
        )

    def test_mtls_rejects_no_client_cert(self, mtls_server_info):
        """Server rejects a client that does not present a certificate."""
        env = _tls_env(
            mtls_server_info["port"],
            mtls_server_info["cert_dir"],
            mtls=False,
        )
        result = _run_in_subprocess(
            _impl_lp_solve_cpu_only, env=env, timeout=30
        )
        assert result.returncode != 0, (
            "Expected failure when connecting without client cert"
        )


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ALLOWED_ENTRIES = {
        "_impl_lp_solve_cpu_only": _impl_lp_solve_cpu_only,
        "_impl_lp_dual_solution_cpu_only": _impl_lp_dual_solution_cpu_only,
        "_impl_mip_solve_cpu_only": _impl_mip_solve_cpu_only,
        "_impl_warmstart_cpu_only": _impl_warmstart_cpu_only,
    }
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in _ALLOWED_ENTRIES:
        print(f"Unknown entry point: {name!r}", file=sys.stderr)
        print(
            f"Available: {', '.join(sorted(_ALLOWED_ENTRIES))}",
            file=sys.stderr,
        )
        sys.exit(1)
    _ALLOWED_ENTRIES[name]()
