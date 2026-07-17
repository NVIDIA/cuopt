#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regenerate fern/docs/scripts/cuopt-install-version.js from the VERSION file.

Runs as a pre-commit hook whenever VERSION changes, keeping the Fern install
selector widget in sync without requiring cuopt_server to be importable.
"""

from pathlib import Path


def main():
    repo_root = Path(__file__).parent.parent.parent
    version = (repo_root / "VERSION").read_text().strip()
    if not version:
        raise ValueError("VERSION file is empty")

    parts = version.split(".")
    major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    pip_ver = f"{major}.{minor}"
    conda_ver = f"{major:02d}.{minor:02d}"

    out = repo_root / "fern/docs/scripts/cuopt-install-version.js"
    out.write_text(
        f'window.CUOPT_INSTALL_VERSION = {{"conda": "{conda_ver}", "pip": "{pip_ver}"}};\n'
    )
    print(
        f"Updated {out.relative_to(repo_root)} (conda={conda_ver}, pip={pip_ver})"
    )


if __name__ == "__main__":
    main()
