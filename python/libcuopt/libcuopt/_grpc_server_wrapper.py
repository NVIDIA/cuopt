# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def main():
    """
    This connects to the gRPC server binary situated under libcuopt/bin folder.
    """
    server_path = os.path.join(
        os.path.dirname(__file__), "bin", "cuopt_grpc_server"
    )
    sys.exit(subprocess.call([server_path] + sys.argv[1:]))
