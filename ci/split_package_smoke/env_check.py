#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helpers for the split-package smoke tests, called from the ci/test_*_split_packages.sh scripts.

Commands:
  conda-packages <env_prefix> <expected_name>...   Fail unless exactly these cuOpt conda packages are installed
  wheel-packages <expected_name>...                Fail unless exactly these libcuopt wheels are installed
  cmake-prefixes                                   Print the CMAKE_PREFIX_PATH of the installed wheels
"""

import importlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys


def installed_conda_packages(prefix):
    """Return the names of the cuOpt packages installed in the conda environment at prefix."""
    out = subprocess.run(
        [
            os.environ.get("CONDA_EXE", "conda"),
            "list",
            "--prefix",
            prefix,
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        p["name"]
        for p in json.loads(out)
        if p["name"].startswith(("libcuopt", "cuopt"))
    }


def installed_wheels():
    """Return the names of the libcuopt wheels installed in the current environment."""
    names = (
        d.metadata["Name"].lower().replace("_", "-")
        for d in metadata.distributions()
    )
    return {n for n in names if n.startswith("libcuopt")}


def check(installed, expected):
    """Exit with an error unless the installed package names are exactly the expected ones."""
    if installed != set(expected):
        print(
            f"installed: {sorted(installed)}, expected: {sorted(expected)}",
            file=sys.stderr,
        )
        sys.exit(1)


def cmake_prefixes():
    """Return the CMake prefix paths of the installed wheels, joined with ";"."""
    return ";".join(
        os.path.dirname(
            importlib.import_module(ep.value.split(":")[0]).__file__
        )
        for ep in metadata.entry_points(group="cmake.prefix")
    )


def main():
    """Run the command named in argv."""
    command, args = sys.argv[1], sys.argv[2:]
    if command == "conda-packages":
        check(installed_conda_packages(args[0]), args[1:])
    elif command == "wheel-packages":
        check(installed_wheels(), args)
    elif command == "cmake-prefixes":
        print(cmake_prefixes())
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
