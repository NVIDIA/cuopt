# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys


def main():
    """
    This connects to cli binary which situated under libcuopt/bin folder
    """
    cli_path = os.path.join(os.path.dirname(__file__), "bin", "cuopt_cli")
    sys.exit(subprocess.call([cli_path] + sys.argv[1:]))
