# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess


def test_cuopt_grpc_server_on_path():
    assert shutil.which("cuopt_grpc_server") is not None, (
        "cuopt_grpc_server should be on PATH after installing cuopt-server"
    )


def test_cuopt_grpc_server_help():
    result = subprocess.run(
        ["cuopt_grpc_server", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"cuopt_grpc_server --help failed (rc={result.returncode}): {result.stderr}"
    )
    assert "cuopt_grpc_server" in result.stdout, (
        f"Expected 'cuopt_grpc_server' in --help output, got: {result.stdout}"
    )
