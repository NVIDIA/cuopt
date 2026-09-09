# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys


def main():
    """
    This connects to the gRPC server binary situated under libcuopt/bin folder.

    execv replaces this process rather than spawning a child. Spawning leaves
    a Python parent that waits on the server but forwards nothing to it, so a
    signal sent to the console script's pid kills only the wrapper: the server
    and its GPU workers survive, orphaned and still holding the listen port,
    and the shutdown path that cancels jobs and reaps workers never runs.
    """
    server_path = os.path.join(
        os.path.dirname(__file__), "bin", "cuopt_grpc_server"
    )
    os.execv(server_path, [server_path] + sys.argv[1:])
