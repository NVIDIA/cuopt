# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import NoReturn


def main() -> NoReturn:
    """Exec the cuopt_cli binary, forwarding this process's arguments to it.

    Never returns: execv replaces the process image. Raises OSError if the
    binary is missing or not executable.

    This connects to cli binary which situated under libcuopt_mathopt/bin folder

    execv replaces this process rather than spawning a child, so signals sent
    to the console script's pid reach the solver directly instead of stopping
    at a Python parent that forwards nothing.
    """
    cli_path = os.path.join(os.path.dirname(__file__), "bin", "cuopt_cli")
    os.execv(cli_path, [cli_path] + sys.argv[1:])
