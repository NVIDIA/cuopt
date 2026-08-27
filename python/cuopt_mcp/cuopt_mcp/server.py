# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server exposing cuOpt LP/MILP solves over the gRPC backend.

Runs as a stdio subprocess of an MCP client, holding a gRPC channel to
``cuopt_grpc_server``. No HTTP is involved and the host needs no GPU — the
solve happens wherever the gRPC server runs.

stdout carries the JSON-RPC stream, so every diagnostic goes to stderr; a
stray ``print()`` here corrupts the protocol.
"""

import logging
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import tools
from .client import CuOptMCPError

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s cuopt-mcp %(levelname)s %(message)s",
)

server = MCPServer(
    name="cuopt",
    instructions=(
        "Solve linear and mixed-integer programs with NVIDIA cuOpt on GPU. "
        "Solves are asynchronous: cuopt_solve_lp / cuopt_solve_milp return a "
        "job_id immediately, then poll cuopt_status and fetch cuopt_result. "
        "Call cuopt_list_settings to discover solver parameters before "
        "passing a settings object. "
        "This server is a client, not a solver: it needs a running "
        "cuopt_grpc_server and never starts one. Call cuopt_health first to "
        "see the configured host/port and whether it answers. If it does "
        "not, check whether a server is already running before starting "
        "another — two servers can end up sharing a port."
    ),
)


def _guard(fn, /, **kwargs) -> dict[str, Any]:
    try:
        return fn(**kwargs)
    except CuOptMCPError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}


@server.tool(structured_output=True)
def cuopt_health() -> dict[str, Any]:
    """Report the configured gRPC target and whether it answers.

    Takes no arguments. Returns host, port, tls, and reachable — plus, when
    unreachable, the error and what to check. Worth calling before building
    a model, since every other tool fails only after the model exists.

    This MCP server never starts or stops cuopt_grpc_server. If the target
    is unreachable, check for a server that is already running before
    starting one.
    """
    return _guard(tools.health)


@server.tool(structured_output=True)
def cuopt_solve_lp(
    problem_path: str | None = None,
    problem: dict | None = None,
    settings: dict | None = None,
) -> dict[str, Any]:
    """Submit a linear program to cuOpt and return a job handle immediately.

    Give the model exactly one of two ways:

    problem_path: path to an MPS, QPS, or LP file readable by this process.
    problem: the model as plain JSON arrays, with no file involved:
        objective: cost per variable (its length defines the column count)
        constraint_matrix: {"rows": [...], "cols": [...], "values": [...]}
            COO triplets, or {"offsets", "indices", "values"} for CSR
        constraint_lower_bounds / constraint_upper_bounds: one per row,
            defaulting to -inf / +inf. Set both equal for an equality row.
            Their length fixes the row count, so a trailing row with all
            zero coefficients is kept rather than inferred away.
        variable_lower_bounds / variable_upper_bounds: default 0 / +inf
        variable_names: labels echoed back by cuopt_result
        maximize: true to maximise (default false)
        objective_offset: constant added to the objective

    Bounds: use null for an unbounded side, since JSON has no infinity
    literal. A magnitude of 1e30 or more is also read as infinite — left
    finite, such a bound can make the solver return a constraint-violating
    point reported as Optimal. Repeated matrix cells are summed.

    settings: optional PDLP solver settings, e.g. {"time_limit": 60,
        "method": "Barrier"}. Call cuopt_list_settings("pdlp_settings") for
        the full list with descriptions and defaults. Omit any setting to
        keep the cuOpt default.

    Returns a job_id. The solve runs asynchronously — poll cuopt_status,
    then call cuopt_result.
    """
    return _guard(
        tools.submit,
        problem_path=problem_path,
        problem=problem,
        kind="pdlp_settings",
        settings=settings,
    )


@server.tool(structured_output=True)
def cuopt_solve_milp(
    problem_path: str | None = None,
    problem: dict | None = None,
    settings: dict | None = None,
) -> dict[str, Any]:
    """Submit a mixed-integer program to cuOpt and return a job handle.

    Give the model exactly one of two ways:

    problem_path: path to an MPS file containing integer variables.
    problem: the model as plain JSON arrays. Prefer this over MPS for
        integer models: an integer column in MPS that has no explicit bound
        entry silently defaults to [0, 1], which turns an ordinary model
        infeasible for no visible reason. variable_types carries
        integrality without touching bounds. Keys — the same set
        cuopt_solve_lp takes, plus variable_types, repeated here because a
        caller may hold this tool without that one:
        objective: cost per variable (its length defines the column count)
        constraint_matrix: {"rows": [...], "cols": [...], "values": [...]}
            COO triplets, or {"offsets", "indices", "values"} for CSR
        constraint_lower_bounds / constraint_upper_bounds: one per row,
            defaulting to -inf / +inf. Set both equal for an equality row.
            Their length fixes the row count, so a trailing row with all
            zero coefficients is kept rather than inferred away.
        variable_lower_bounds / variable_upper_bounds: default 0 / +inf
        variable_types: per-variable "I" (integer) or "C" (continuous)
        variable_names: labels echoed back by cuopt_result
        maximize: true to maximise (default false)
        objective_offset: constant added to the objective

    Bounds: use null for an unbounded side, since JSON has no infinity
    literal. A magnitude of 1e30 or more is also read as infinite — left
    finite, such a bound can make the solver return a constraint-violating
    point reported as Optimal. Repeated matrix cells are summed.

    settings: optional MIP solver settings, e.g. {"time_limit": 300,
        "relative_mip_gap": 0.01}. Call cuopt_list_settings("mip_settings")
        for the full list — parameter names are easy to guess wrong
        (relative_mip_gap, not mip_relative_gap).

    Returns a job_id. Use cuopt_incumbents to watch the objective improve
    and cuopt_cancel to stop early once it is good enough.
    """
    return _guard(
        tools.submit,
        problem_path=problem_path,
        problem=problem,
        kind="mip_settings",
        settings=settings,
    )


@server.tool(structured_output=True)
def cuopt_status(job_id: str) -> dict[str, Any]:
    """Report whether a cuOpt job is queued, running, or finished.

    Cheap to call repeatedly. Returns terminal=true once the job has
    reached COMPLETED, FAILED, CANCELLED, or NOT_FOUND.
    """
    return _guard(tools.status, job_id=job_id)


@server.tool(structured_output=True)
def cuopt_result(
    job_id: str,
    names_from: str | None = None,
    variables: list | None = None,
    nonzero_only: bool = False,
    limit: int = tools.INLINE_SOLUTION_LIMIT,
) -> dict[str, Any]:
    """Fetch the solution for a finished cuOpt job.

    Always returns the termination status, objective, and solve time.
    Variable values are shaped to stay readable:

    names_from: path to the problem file, to key values by variable name
        rather than column index. Pass the "source" returned by the solve.
    variables: fetch only these named variables.
    nonzero_only: return only variables with a non-zero value — usually
        what matters for a MILP.
    limit: maximum values returned inline. Beyond this the full solution is
        written to a file and its path returned instead.
    """
    return _guard(
        tools.result,
        job_id=job_id,
        names_from=names_from,
        variables=variables,
        nonzero_only=nonzero_only,
        limit=limit,
    )


@server.tool(structured_output=True)
def cuopt_incumbents(job_id: str, from_index: int = 0) -> dict[str, Any]:
    """Return improving MILP solutions found so far, oldest first.

    Use the returned next_index on the following call to fetch only new
    incumbents. A flat objective across several calls means the solver has
    plateaued and cuopt_cancel may be worthwhile.
    """
    return _guard(tools.incumbents, job_id=job_id, from_index=from_index)


@server.tool(structured_output=True)
def cuopt_logs(
    job_id: str, from_byte: int = 0, tail_lines: int = 100
) -> dict[str, Any]:
    """Return recent solver log lines for a job, for diagnosing a slow solve."""
    return _guard(
        tools.logs, job_id=job_id, from_byte=from_byte, tail_lines=tail_lines
    )


@server.tool(structured_output=True)
def cuopt_cancel(job_id: str) -> dict[str, Any]:
    """Stop a running cuOpt job. Any incumbent found so far remains fetchable."""
    return _guard(tools.cancel, job_id=job_id)


@server.tool(structured_output=True)
def cuopt_list_settings(kind: str, name: str | None = None) -> dict[str, Any]:
    """List cuOpt solver settings with descriptions, types, and defaults.

    kind: "pdlp_settings" for LP, "mip_settings" for MILP.
    name: a single parameter to describe in full, instead of listing names.

    The catalogue is generated from cuOpt's field registry, so it always
    matches the solver build being talked to.
    """
    return _guard(tools.list_settings, kind=kind, name=name)


def main() -> None:
    host, port = __import__(
        "cuopt_mcp.client", fromlist=["endpoint"]
    ).endpoint()
    logging.info("cuopt-mcp starting; gRPC target %s:%s", host, port)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
