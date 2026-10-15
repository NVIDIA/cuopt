# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server exposing cuOpt LP/MILP solves over the gRPC backend.

Runs as a stdio subprocess of an MCP client, holding a gRPC channel to
``cuopt_grpc_server``. No HTTP application endpoint is exposed, and the
host needs no GPU — the solve happens wherever the gRPC server runs.

stdout carries the JSON-RPC stream, so every diagnostic goes to stderr; a
stray ``print()`` here corrupts the protocol.
"""

import logging
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import tools
from .client import CuOptMCPError, endpoint, redact_paths

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
        "passing a settings object."
    ),
)


def _guard(fn, /, **kwargs) -> dict[str, Any]:
    """Call a tools.* function, turning a caller-facing error into a normal
    return value instead of an MCP protocol exception.

    Every tool below returns this dict's shape on success; on failure it's
    ``{"error": <message safe to show the model>}`` instead of a traceback,
    which the model can act on inline instead of the call simply failing.
    Paths are redacted here so every raise site doesn't have to.

    An exception of any other type is logged (with traceback) to stderr
    and reduced to a generic message instead of str(exc): unlike
    CuOptMCPError/ValueError, that text was never vetted as safe to
    return, and letting it through here would bypass redact_paths.
    """
    try:
        return fn(**kwargs)
    except (CuOptMCPError, ValueError) as exc:
        return {"error": redact_paths(str(exc))}
    except Exception:
        logging.exception("unexpected error in %s", fn.__qualname__)
        return {"error": "internal error -- see server logs"}


@server.tool(structured_output=True)
def cuopt_solve_lp(
    problem_path: str, settings: dict | None = None
) -> dict[str, Any]:
    """Submit a linear program to cuOpt and return a job handle immediately.

    problem_path: path to an MPS, QPS, or LP file readable by this process.
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
        kind="pdlp_settings",
        settings=settings,
    )


@server.tool(structured_output=True)
def cuopt_solve_milp(
    problem_path: str,
    settings: dict | None = None,
    track_incumbents: bool = False,
) -> dict[str, Any]:
    """Submit a mixed-integer program to cuOpt and return a job handle.

    problem_path: path to an MPS or LP file declaring integer/binary
        variables (LP: Generals/Binaries sections).
    settings: optional MIP solver settings, e.g. {"time_limit": 300,
        "relative_mip_gap": 0.01}. Call cuopt_list_settings("mip_settings")
        for the full list.
    track_incumbents: set True to make cuopt_incumbents useful for this
        job. Off by default -- it costs extra server-side work and network
        transfer per incumbent found, worth paying only if you'll poll it.

    Returns a job_id. Use cuopt_cancel to stop early once the result is
    good enough.
    """
    return _guard(
        tools.submit,
        problem_path=problem_path,
        kind="mip_settings",
        settings=settings,
        track_incumbents=track_incumbents,
    )


@server.tool(structured_output=True)
def cuopt_status(job_id: str) -> dict[str, Any]:
    """Report whether a cuOpt job is queued, running, or finished.

    Cheap to call repeatedly. Returns terminal=true once the job has
    reached COMPLETED, FAILED, CANCELLED, or NOT_FOUND. On failure, returns
    ``{"error": <message>}`` instead (see ``_guard``).
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
    limit: maximum values returned inline. Beyond this the selected values
        (respecting nonzero_only) are written to a file and its path
        returned instead.

    On failure, returns ``{"error": <message>}`` instead (see ``_guard``).
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

    Requires cuopt_solve_milp's track_incumbents=True for this job, or
    this always comes back empty. Use the returned next_index on the
    following call to fetch only new incumbents. A flat objective across
    several calls means the solver has plateaued and cuopt_cancel may be
    worthwhile. On failure, returns ``{"error": <message>}`` instead
    (see ``_guard``).
    """
    return _guard(tools.incumbents, job_id=job_id, from_index=from_index)


@server.tool(structured_output=True)
def cuopt_logs(
    job_id: str, from_byte: int = 0, tail_lines: int = 100
) -> dict[str, Any]:
    """Return solver log lines for a finished job, for diagnosing a failure
    or an unexpected result after the fact.

    Only works once the job has reached a terminal state (poll
    cuopt_status first) — a live tail of a still-running job isn't
    available through this tool yet. For a CANCELLED job, expect an empty
    or missing log: the server deletes it as part of cancelling, unlike
    COMPLETED/FAILED.

    job_id: the job to fetch logs for.
    from_byte: resume from this byte offset — pass back the next_byte from
        a prior call to fetch only what's new since then.
    tail_lines: keep only the last this many lines of the fetched text;
        must be between 1 and tools.MAX_TAIL_LINES.

    Returns lines, truncated (whether more preceded the kept lines), and
    next_byte for the following call. On failure, or on an out-of-range
    tail_lines, returns ``{"error": <message>}`` instead (see ``_guard``).
    """
    return _guard(
        tools.logs, job_id=job_id, from_byte=from_byte, tail_lines=tail_lines
    )


@server.tool(structured_output=True)
def cuopt_cancel(job_id: str) -> dict[str, Any]:
    """Stop a running cuOpt job. Any incumbent found so far remains fetchable.

    job_id: the job to cancel. Cancelling a job that has already reached
    COMPLETED or FAILED returns ``{"error": <message>}`` (see ``_guard``)
    rather than succeeding silently.
    """
    return _guard(tools.cancel, job_id=job_id)


@server.tool(structured_output=True)
def cuopt_delete(job_id: str) -> dict[str, Any]:
    """Release a finished job's server-side state (solution, logs,
    incumbents). Cancels first if it is still running.

    job_id: the job to delete. Call this once its result is no longer
    needed, so cuopt_grpc_server doesn't accumulate state indefinitely.
    On failure, returns ``{"error": <message>}`` instead (see ``_guard``).
    """
    return _guard(tools.delete, job_id=job_id)


@server.tool(structured_output=True)
def cuopt_list_settings(kind: str, name: str | None = None) -> dict[str, Any]:
    """List cuOpt solver settings with descriptions, types, and defaults.

    kind: "pdlp_settings" for LP, "mip_settings" for MILP.
    name: a single parameter to describe in full, instead of listing names.

    The catalogue is generated from cuOpt's field registry, so it always
    matches the solver build being talked to. On an unknown kind or name,
    returns ``{"error": <message>}`` instead (see ``_guard``).
    """
    return _guard(tools.list_settings, kind=kind, name=name)


def main() -> None:
    """Run the MCP server over stdio.

    Blocks until the client disconnects or the process is killed. Logs the
    configured gRPC target to stderr on startup; raises whatever
    ``server.run`` raises on a fatal transport failure (stdout is reserved
    for the JSON-RPC stream, so nothing here writes there).
    """
    host, port = endpoint()
    logging.info("cuopt-mcp starting; gRPC target %s:%s", host, port)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
