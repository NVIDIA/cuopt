# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server for NVIDIA cuOpt."""

__all__ = ["main"]


def main():
    """Console-script entry point (``cuopt-mcp``)."""
    from .server import main as _main

    _main()
