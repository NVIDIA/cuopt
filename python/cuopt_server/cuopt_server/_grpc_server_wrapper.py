# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def main():
    """
    Wrapper that launches the cuopt_grpc_server binary from the libcuopt package.
    """
    import libcuopt

    server_path = os.path.join(
        os.path.dirname(libcuopt.__file__), "bin", "cuopt_grpc_server"
    )
    sys.exit(subprocess.call([server_path] + sys.argv[1:]))
