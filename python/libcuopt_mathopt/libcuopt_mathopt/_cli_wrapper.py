# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys


def main():
    """
    This connects to cli binary which situated under libcuopt/bin folder

    execv replaces this process rather than spawning a child, so signals sent
    to the console script's pid reach the solver directly instead of stopping
    at a Python parent that forwards nothing.
    """
    cli_path = os.path.join(os.path.dirname(__file__), "bin", "cuopt_cli")
    os.execv(cli_path, [cli_path] + sys.argv[1:])
