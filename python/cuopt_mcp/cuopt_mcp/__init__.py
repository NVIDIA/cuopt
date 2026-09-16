# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server for NVIDIA cuOpt."""

__all__ = ["main"]


def main() -> None:
    """Console-script entry point (``cuopt-mcp``).

    Runs the MCP server over stdio until the client disconnects or the
    process is killed; does not return until then.
    """
    from .server import main as _main

    _main()
