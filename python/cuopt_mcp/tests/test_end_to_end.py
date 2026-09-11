# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test: a real MCP client driving a real cuOpt solve.

Spawns the MCP server as a stdio subprocess exactly as an MCP client would,
speaks the MCP protocol to it, and lets it forward the solve to a live
cuopt_grpc_server. Nothing is mocked.

Set CUOPT_TEST_GRPC_PORT to point at a running server; the test skips if no
server is reachable.
"""

import os
import textwrap

import pytest

mcp = pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters, stdio_client  # noqa: E402

PORT = os.environ.get("CUOPT_TEST_GRPC_PORT")

pytestmark = pytest.mark.skipif(
    not PORT, reason="CUOPT_TEST_GRPC_PORT not set; no live cuopt_grpc_server"
)

# minimize  x + y   s.t.  x + y >= 10,  x <= 8,  y <= 8
# optimum: 10 with x + y == 10
LP_MPS = textwrap.dedent(
    """\
    NAME          TESTLP
    ROWS
     N  COST
     G  LIM1
    COLUMNS
        X         COST      1.0        LIM1      1.0
        Y         COST      1.0        LIM1      1.0
    RHS
        RHS       LIM1      10.0
    BOUNDS
     UP BND       X         8.0
     UP BND       Y         8.0
    ENDATA
    """
)


@pytest.fixture
def mps_file(tmp_path):
    path = tmp_path / "testlp.mps"
    path.write_text(LP_MPS)
    return str(path)


@pytest.fixture
async def session(tmp_path):
    params = StdioServerParameters(
        command="cuopt-mcp",
        env={
            **os.environ,
            "CUOPT_REMOTE_HOST": "localhost",
            "CUOPT_REMOTE_PORT": PORT,
            "CUOPT_MCP_SOLUTION_DIR": str(tmp_path / "solutions"),
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            yield client


async def _call(session, tool, /, **args):
    """Positional-only params so a tool argument named `name` cannot shadow them."""
    result = await session.call_tool(tool, args)
    assert not result.is_error, result.content
    return result.structured_content


@pytest.mark.anyio
async def test_tools_are_discoverable(session):
    listed = await session.list_tools()
    names = {t.name for t in listed.tools}
    assert {"cuopt_solve_lp", "cuopt_status", "cuopt_result"} <= names


@pytest.mark.anyio
async def test_settings_catalogue_reaches_the_client(session):
    detail = await _call(
        session,
        "cuopt_list_settings",
        kind="pdlp_settings",
        name="pdlp_solver_mode",
    )
    assert "Stable3" in detail["enum"]
    assert "Default: Stable3" in detail["description"]


@pytest.mark.anyio
async def test_solve_lp_end_to_end(session, mps_file):
    """Submit, poll, and fetch a named solution through the MCP protocol."""
    submitted = await _call(
        session,
        "cuopt_solve_lp",
        problem_path=mps_file,
        settings={"time_limit": 30.0},
    )
    assert "error" not in submitted, submitted
    job_id = submitted["job_id"]
    assert submitted["num_variables"] == 2

    for _ in range(120):
        state = await _call(session, "cuopt_status", job_id=job_id)
        if state["terminal"]:
            break
        await __import__("asyncio").sleep(0.5)
    assert state["status"] == "COMPLETED", state

    solved = await _call(
        session, "cuopt_result", job_id=job_id, names_from=mps_file
    )
    assert solved["ready"] is True
    assert solved["primal_objective"] == pytest.approx(10.0, abs=1e-4)
    total = sum(solved["variables"].values())
    assert total == pytest.approx(10.0, abs=1e-4)


@pytest.mark.anyio
async def test_invalid_setting_is_rejected_before_submission(
    session, mps_file
):
    """A typo must come back as a named error, not a silently ignored field."""
    out = await _call(
        session,
        "cuopt_solve_lp",
        problem_path=mps_file,
        settings={"time_limt": 5.0},
    )
    assert "error" in out
    assert "did you mean time_limit" in out["error"]


@pytest.mark.anyio
async def test_missing_file_reports_cleanly(session):
    out = await _call(session, "cuopt_solve_lp", problem_path="/no/such.mps")
    assert "problem file not found" in out["error"]


@pytest.mark.anyio
async def test_enum_setting_is_accepted_by_name(session, mps_file):
    """Regression: settings={"method": "Barrier"} must reach the solver.

    cuOpt's set_parameter takes an integer for enum settings, so passing the
    readable name straight through failed with "value Barrier is not an
    integer".
    """
    out = await _call(
        session,
        "cuopt_solve_lp",
        problem_path=mps_file,
        settings={"method": "Barrier", "time_limit": 30.0},
    )
    assert "error" not in out, out
    assert out["job_id"]


@pytest.mark.anyio
async def test_termination_status_is_readable(session, mps_file):
    """Regression: status must be a name, not the IntEnum's bare number."""
    sub = await _call(
        session,
        "cuopt_solve_lp",
        problem_path=mps_file,
        settings={"time_limit": 30.0},
    )
    while not (await _call(session, "cuopt_status", job_id=sub["job_id"]))[
        "terminal"
    ]:
        await __import__("asyncio").sleep(0.3)
    out = await _call(session, "cuopt_result", job_id=sub["job_id"])
    assert out["termination_status"] == "Optimal"
    assert out["termination_status_code"] == 1
