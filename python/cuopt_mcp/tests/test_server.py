# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""_guard's error-shaping behavior, isolated from the MCP protocol."""

import inspect

from cuopt_mcp.client import CuOptMCPError
from cuopt_mcp.server import _guard, cuopt_solve_lp, cuopt_solve_milp


def test_problem_is_keyword_only_on_solve_tools():
    """``problem`` was inserted as an alternative to ``problem_path`` after
    these tools' 2/3-positional-arg signatures had already shipped
    (#1916) -- a positional (path, settings) call must still bind
    settings to settings, not silently to problem.
    """
    for fn in (cuopt_solve_lp, cuopt_solve_milp):
        params = inspect.signature(fn).parameters
        assert params["problem"].kind == inspect.Parameter.KEYWORD_ONLY


def test_guard_redacts_paths_in_cuoptmcperror():
    def raiser():
        raise CuOptMCPError("/home/alice/.cuopt-mcp/solutions exists")

    out = _guard(raiser)
    assert "/home/alice" not in out["error"]
    assert "[redacted path]" in out["error"]


def test_guard_redacts_paths_in_value_error():
    def raiser():
        raise ValueError("bad value for /etc/cuopt/secret.conf")

    out = _guard(raiser)
    assert "/etc/cuopt" not in out["error"]


def test_guard_passes_through_on_success():
    assert _guard(lambda: {"ok": True}) == {"ok": True}


def test_guard_reduces_unexpected_exceptions_to_a_generic_message():
    """Not CuOptMCPError/ValueError, so its raw text was never vetted as
    safe -- found via a real ImportError escaping to an MCP caller during
    live testing.
    """

    def raiser():
        raise ImportError("/some/internal/path: undefined symbol: secret_abi")

    out = _guard(raiser)
    assert "/some/internal/path" not in out["error"]
    assert "secret_abi" not in out["error"]
    assert out["error"] == "internal error -- see server logs"
