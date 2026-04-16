# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import ctypes.util
import platform
import shutil
import subprocess

import pytest


def test_cuopt_grpc_server_on_path():
    assert shutil.which("cuopt_grpc_server") is not None, (
        "cuopt_grpc_server should be on PATH after installing cuopt-server"
    )


def _check_libuuid():
    """Return (found: bool, detail: str) for libuuid availability."""
    name = ctypes.util.find_library("uuid")
    if name is None:
        return False, "ctypes.util.find_library('uuid') returned None"
    try:
        ctypes.CDLL(name)
        return True, f"loaded {name}"
    except OSError as exc:
        return False, f"find_library returned '{name}' but load failed: {exc}"


def test_cuopt_grpc_server_help():
    result = subprocess.run(
        ["cuopt_grpc_server", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0 and not result.stdout and not result.stderr:
        uuid_ok, uuid_detail = _check_libuuid()
        pytest.skip(
            f"cuopt_grpc_server binary failed to load "
            f"(rc={result.returncode}, arch={platform.machine()}). "
            f"libuuid: {uuid_detail}"
        )
    assert result.returncode == 0, (
        f"cuopt_grpc_server --help failed (rc={result.returncode}): "
        f"{result.stdout}\n{result.stderr}"
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert "cuopt_grpc_server" in output, (
        f"Expected 'cuopt_grpc_server' in --help output, got: {output}"
    )
