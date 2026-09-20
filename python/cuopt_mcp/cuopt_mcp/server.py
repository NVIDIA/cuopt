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

from . import routing, tools
from .client import CuOptMCPError, endpoint, redact_paths

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s cuopt-mcp %(levelname)s %(message)s",
)

server = MCPServer(
    name="cuopt",
    instructions=(
        "Solve linear programs, mixed-integer programs, and vehicle "
        "routing problems with NVIDIA cuOpt on GPU. Solves are "
        "asynchronous: cuopt_solve_lp / cuopt_solve_milp / cuopt_solve_vrp "
        "return a job_id immediately, then poll cuopt_status and fetch "
        "cuopt_result (LP/MILP) or cuopt_vrp_result (VRP) -- cuopt_status/"
        "cuopt_cancel/cuopt_delete work for a job_id from any of the three. "
        "Call cuopt_list_settings to discover LP/MILP solver parameters "
        "before passing a settings object. "
        "This server is a client, not a solver: it needs a running "
        "cuopt_grpc_server and never starts one. Call cuopt_health first to "
        "see the configured host/port and whether it answers. If it does "
        "not, check whether a server is already running before starting "
        "another — two servers can end up sharing a port."
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
    settings: dict | None = None,
    *,
    problem: dict | None = None,
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
    settings: dict | None = None,
    track_incumbents: bool = False,
    *,
    problem: dict | None = None,
) -> dict[str, Any]:
    """Submit a mixed-integer program to cuOpt and return a job handle.

    Give the model exactly one of two ways:

    problem_path: path to an MPS or LP file declaring integer/binary
        variables (LP: Generals/Binaries sections; MPS: an integer column
        with no explicit bound entry silently defaults to [0, 1], which
        turns an ordinary model infeasible for no visible reason).
    problem: the model as plain JSON arrays. Prefer this over MPS for
        integer models: variable_types carries integrality without
        touching bounds. Keys — the same set cuopt_solve_lp takes, plus
        variable_types, repeated here because a caller may hold this tool
        without that one:
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
    track_incumbents: set True to make cuopt_incumbents useful for this
        job. Off by default -- it costs extra server-side work and network
        transfer per incumbent found, worth paying only if you'll poll it.

    Returns a job_id. Use cuopt_cancel to stop early once the result is
    good enough.
    """
    return _guard(
        tools.submit,
        problem_path=problem_path,
        problem=problem,
        kind="mip_settings",
        settings=settings,
        track_incumbents=track_incumbents,
    )


@server.tool(structured_output=True)
def cuopt_solve_vrp(
    problem: dict, settings: dict | None = None
) -> dict[str, Any]:
    """Submit a vehicle routing problem (VRP/PDP) to cuOpt and return a job
    handle immediately.

    problem: the model as plain JSON arrays, mirroring RoutingProblem in
        cuopt_routing.proto field for field. Required:
        n_locations, fleet_size: sizes (locations include vehicle start/end
            points; n_orders defaults to n_locations if omitted).
        cost_matrices: [{"values": [[...]], "vehicle_type": 0}] -- one
            n_locations x n_locations matrix per vehicle_type (heterogeneous
            fleets use more than one). At least one required.
        Optional, by area:
        transit_time_matrices: same shape as cost_matrices, used for time-
            window feasibility instead of cost_matrices when set.
        vehicle_locations: {"start": [...], "end": [...]}, vehicle_types,
            vehicle_time_windows: {"earliest": [...], "latest": [...]},
            drop_return_trips, skip_first_trips, vehicle_max_costs,
            vehicle_max_times, vehicle_fixed_costs -- each length fleet_size.
        order_locations, order_prizes -- each length n_orders.
        order_time_windows: {"earliest": [...], "latest": [...]}, length
            n_orders (per order, not per location).
        order_service_times: [{"service_times": [...], "vehicle_id": -1}] --
            vehicle_id -1 (default) sets the fallback for all vehicles.
        pickup_delivery_pairs: {"pickup": [...], "delivery": [...]} -- order
            indices (positions into order_locations), not location ids.
        capacity_dimensions: [{"name": ..., "demand": [...] (n_orders),
            "capacity": [...] (fleet_size)}] -- one entry per dimension.
        break_locations: allowed break locations (default: any).
        uniform_breaks: [{"earliest": [...], "latest": [...],
            "duration": [...]}] (each length fleet_size) -- fleet-wide
            breaks. Mutually exclusive with vehicle_breaks/
            vehicle_distance_breaks.
        vehicle_breaks: [{"vehicle_id", "earliest", "latest", "duration",
            "locations": [...] (optional)}] -- one entry per break.
        vehicle_distance_breaks: [{"vehicle_id", "distance_min",
            "distance_max", "duration", "locations": [...] (optional)}].
        vehicle_order_match: [{"vehicle_id", "orders": [...]}] -- restricts
            a vehicle to only the given orders.
        order_vehicle_match: [{"order_id", "vehicles": [...]}] -- restricts
            an order to only the given vehicles.
        order_precedence: [{"order_id", "preceding_orders": [...]}].
        objective: {"objectives": [...names...], "weights": [...]}. Names:
            COST, TRAVEL_TIME, VARIANCE_ROUTE_SIZE,
            VARIANCE_ROUTE_SERVICE_TIME, PRIZE, VEHICLE_FIXED_COST,
            DISTANCE_BREAK_COST. Default weight 1.0 for COST and for any
            objective whose matching input (prizes, fixed costs, distance
            breaks) is set; 0.0 otherwise.
        min_vehicles: floor on fleet size used (solution may not be
            optimal when set).
        initial_solutions: {"vehicle_ids", "routes", "sol_offsets": [...],
            "types": [...]} -- types are "Depot"/"Pickup"/"Delivery"/
            "Break".

    settings: optional, e.g. {"time_limit": 30}. Only time_limit,
        verbose_mode (or verbose), and error_logging reach the server.

    Returns a job_id. Use cuopt_vrp_result once cuopt_status reports
    COMPLETED.
    """
    return _guard(routing.submit, problem=problem, settings=settings)


@server.tool(structured_output=True)
def cuopt_vrp_result(
    job_id: str, limit: int = tools.INLINE_SOLUTION_LIMIT
) -> dict[str, Any]:
    """Fetch the solution for a finished VRP job.

    job_id: a job handle previously returned by cuopt_solve_vrp.
    limit: maximum route stops returned inline. Beyond this the full route
        table is written to a file and its path returned instead.

    Returns status (SUCCESS/FAIL/TIMEOUT/EMPTY), total_objective_value,
    objective_values (by name), vehicle_count, and stops -- a flat list of
    {vehicle, location, type, arrival} across all routes, one per visit
    (type is "Depot"/"Pickup"/"Delivery"/"Break"). On FAIL, also carries
    unserviced_orders. On failure, or on an out-of-range limit, returns
    ``{"error": <message>}`` instead (see ``_guard``).
    """
    return _guard(routing.result, job_id=job_id, limit=limit)


@server.tool(structured_output=True)
def cuopt_status(job_id: str) -> dict[str, Any]:
    """Report whether a cuOpt job is queued, running, or finished.

    Works for a job_id from any of cuopt_solve_lp/cuopt_solve_milp/
    cuopt_solve_vrp -- job_id is server-issued from one registry
    regardless of problem type. Cheap to call repeatedly. Returns
    terminal=true once the job has reached COMPLETED, FAILED, CANCELLED,
    or NOT_FOUND. On failure, returns ``{"error": <message>}`` instead
    (see ``_guard``).
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
    nonzero_only: drop values within solver tolerance of zero (not
        necessarily exact zero) — usually what matters for a MILP.
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
    """Stop a running cuOpt job (LP, MILP, or VRP). Any incumbent found so
    far remains fetchable.

    job_id: the job to cancel. Cancelling a job that has already reached
    COMPLETED or FAILED returns ``{"error": <message>}`` (see ``_guard``)
    rather than succeeding silently.
    """
    return _guard(tools.cancel, job_id=job_id)


@server.tool(structured_output=True)
def cuopt_delete(job_id: str) -> dict[str, Any]:
    """Release a finished job's server-side state (solution, logs,
    incumbents). Works for LP, MILP, or VRP jobs. Cancels first if it is
    still running.

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
