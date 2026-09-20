# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VRP (vehicle routing) tools for the cuOpt MCP server.

Separate from tools.py: VRP uses a different gRPC client
(``RoutingClient``), a plain-dict settings object instead of a generated
schema, and a route table instead of a variable vector, so little of the
LP/MIP machinery is shared. cuopt_status/cuopt_cancel/cuopt_delete are
shared, though -- job_id is server-assigned from one registry regardless
of problem category, so those tools already work for a VRP job_id.

A submitted model never touches the GPU on this host: ``routing.DataModel``
records setter calls (numpy/plain-Python arrays) into a serializable form
and only builds the device model when solved locally, which this process
never does -- it only serializes the recorded calls onto the wire.
"""

import json

from .client import (
    CuOptMCPError,
    describe_connection_error,
    get_routing_client,
)
from .tools import INLINE_SOLUTION_LIMIT, _solution_file_path

# node_type_t (cpp/include/cuopt/routing/routing_structures.hpp): plain
# sequential enum, no explicit values. Names match
# cuopt.grpc.client.grpc_client._NODE_TYPE_NAMES exactly (that mapping is
# what add_initial_solutions' `types` argument is actually matched against).
_NODE_TYPES = {"Depot": 0, "Pickup": 1, "Delivery": 2, "Break": 3}
_NODE_TYPE_NAMES = {v: k for k, v in _NODE_TYPES.items()}

# objective_t (same header): also a plain sequential enum. No name mapping
# exists on the client side (unlike node types) because nothing needs one
# for local solving -- built here specifically for this JSON surface.
_OBJECTIVES = {
    "COST": 0,
    "TRAVEL_TIME": 1,
    "VARIANCE_ROUTE_SIZE": 2,
    "VARIANCE_ROUTE_SERVICE_TIME": 3,
    "PRIZE": 4,
    "VEHICLE_FIXED_COST": 5,
    "DISTANCE_BREAK_COST": 6,
}
_OBJECTIVE_NAMES = {v: k for k, v in _OBJECTIVES.items()}

# assignment.SolutionStatus, mirrored here rather than imported: importing
# the real enum pulls the compiled routing wrapper (and cudf) at module
# scope, defeating the lazy-import discipline this package otherwise keeps.
_STATUS_NAMES = {0: "SUCCESS", 1: "FAIL", 2: "TIMEOUT", 3: "EMPTY"}


def _node_type_value(value, where: str) -> str:
    name = str(value).capitalize()
    if name not in _NODE_TYPES:
        raise CuOptMCPError(
            f"{where}: {value!r} is not a node type ({sorted(_NODE_TYPES)})"
        )
    return name


def _objective_value(value) -> int:
    name = str(value).upper()
    if name not in _OBJECTIVES:
        raise CuOptMCPError(
            f"objective {value!r} is not one of {sorted(_OBJECTIVES)}"
        )
    return _OBJECTIVES[name]


def _require(problem: dict, key: str):
    if key not in problem:
        raise CuOptMCPError(f"problem is missing required key {key!r}")
    return problem[key]


def _i32(values):
    import numpy as np

    return np.asarray(values, dtype=np.int32)


def _f32(values):
    import numpy as np

    return np.asarray(values, dtype=np.float32)


def _u8(values):
    import numpy as np

    return np.asarray(values, dtype=np.uint8)


def _bool_arr(values):
    import numpy as np

    return np.asarray(values, dtype=bool)


def _opt_i32(values):
    """Like _i32, but None passes through -- an unset "locations" argument
    means "any location", not an empty array.
    """
    return None if values is None else _i32(values)


def _build_routing_model_from_json(problem: dict):
    """Build a routing.DataModel from plain JSON arrays.

    Every array-like value is converted to a numpy array before reaching a
    setter: the Python-side validators several setters run eagerly
    (validate_matrix, validate_time_windows) explicitly reject plain
    Python lists, unlike the gRPC serialization layer underneath (which
    accepts anything array-like).

    Raises
    ------
        CuOptMCPError: A required key is missing, an array has the wrong
            shape, or a mutually exclusive combination was given.
    """
    import numpy as np

    from cuopt.routing import DataModel

    if not isinstance(problem, dict):
        raise CuOptMCPError("problem must be an object")

    n_locations = int(_require(problem, "n_locations"))
    fleet_size = int(_require(problem, "fleet_size"))
    n_orders = int(problem.get("n_orders", -1))
    dm = DataModel(n_locations, fleet_size, n_orders)

    cost_matrices = _require(problem, "cost_matrices")
    if not cost_matrices:
        raise CuOptMCPError("cost_matrices must have at least one entry")
    for entry in cost_matrices:
        dm.add_cost_matrix(_f32(entry["values"]), entry.get("vehicle_type", 0))
    for entry in problem.get("transit_time_matrices", []):
        dm.add_transit_time_matrix(
            _f32(entry["values"]), entry.get("vehicle_type", 0)
        )

    if "vehicle_types" in problem:
        dm.set_vehicle_types(_u8(problem["vehicle_types"]))
    if "vehicle_locations" in problem:
        vl = problem["vehicle_locations"]
        dm.set_vehicle_locations(_i32(vl["start"]), _i32(vl["end"]))
    if "vehicle_time_windows" in problem:
        vtw = problem["vehicle_time_windows"]
        dm.set_vehicle_time_windows(_i32(vtw["earliest"]), _i32(vtw["latest"]))
    if "drop_return_trips" in problem:
        dm.set_drop_return_trips(_bool_arr(problem["drop_return_trips"]))
    if "skip_first_trips" in problem:
        dm.set_skip_first_trips(_bool_arr(problem["skip_first_trips"]))
    if "vehicle_max_costs" in problem:
        dm.set_vehicle_max_costs(_f32(problem["vehicle_max_costs"]))
    if "vehicle_max_times" in problem:
        dm.set_vehicle_max_times(_f32(problem["vehicle_max_times"]))
    if "vehicle_fixed_costs" in problem:
        dm.set_vehicle_fixed_costs(_f32(problem["vehicle_fixed_costs"]))

    if "order_locations" in problem:
        dm.set_order_locations(_i32(problem["order_locations"]))
    if "order_time_windows" in problem:
        otw = problem["order_time_windows"]
        dm.set_order_time_windows(_i32(otw["earliest"]), _i32(otw["latest"]))
    if "order_prizes" in problem:
        dm.set_order_prizes(_f32(problem["order_prizes"]))
    for entry in problem.get("order_service_times", []):
        dm.set_order_service_times(
            _i32(entry["service_times"]), entry.get("vehicle_id", -1)
        )

    if "pickup_delivery_pairs" in problem:
        pdp = problem["pickup_delivery_pairs"]
        # pickup/delivery are order indices (positions into order_locations
        # etc.), not location ids -- set_order_locations should be called
        # first for these to mean anything, though nothing here enforces it
        # (the underlying setter doesn't either; a mismatch surfaces at
        # solve time, not here).
        dm.set_pickup_delivery_pairs(
            _i32(pdp["pickup"]), _i32(pdp["delivery"])
        )

    for entry in problem.get("capacity_dimensions", []):
        dm.add_capacity_dimension(
            entry["name"], _i32(entry["demand"]), _i32(entry["capacity"])
        )

    if "break_locations" in problem:
        dm.set_break_locations(_i32(problem["break_locations"]))
    uniform_breaks = problem.get("uniform_breaks", [])
    vehicle_breaks = problem.get("vehicle_breaks", [])
    vehicle_distance_breaks = problem.get("vehicle_distance_breaks", [])
    if uniform_breaks and (vehicle_breaks or vehicle_distance_breaks):
        raise CuOptMCPError(
            "uniform_breaks and vehicle_breaks/vehicle_distance_breaks are "
            "mutually exclusive -- fleet-wide breaks or per-vehicle breaks, "
            "not both"
        )
    for entry in uniform_breaks:
        dm.add_break_dimension(
            _i32(entry["earliest"]),
            _i32(entry["latest"]),
            _i32(entry["duration"]),
        )
    for entry in vehicle_breaks:
        dm.add_vehicle_break(
            entry["vehicle_id"],
            entry["earliest"],
            entry["latest"],
            entry["duration"],
            _opt_i32(entry.get("locations")),
        )
    for entry in vehicle_distance_breaks:
        dm.add_vehicle_distance_break(
            entry["vehicle_id"],
            entry["distance_min"],
            entry["distance_max"],
            entry["duration"],
            _opt_i32(entry.get("locations")),
        )

    for entry in problem.get("vehicle_order_match", []):
        dm.add_vehicle_order_match(entry["vehicle_id"], _i32(entry["orders"]))
    for entry in problem.get("order_vehicle_match", []):
        dm.add_order_vehicle_match(entry["order_id"], _i32(entry["vehicles"]))
    for entry in problem.get("order_precedence", []):
        # add_order_precedence has no Python-side validation at all (not
        # declared in vehicle_routing.py, only the compiled wrapper), so a
        # bad order_id/preceding_orders value isn't caught until solve time.
        dm.add_order_precedence(
            entry["order_id"], _i32(entry["preceding_orders"])
        )

    if "objective" in problem:
        obj = problem["objective"]
        objectives = _i32([_objective_value(o) for o in obj["objectives"]])
        dm.set_objective_function(objectives, _f32(obj["weights"]))
    if "min_vehicles" in problem:
        dm.set_min_vehicles(int(problem["min_vehicles"]))
    if "initial_solutions" in problem:
        init = problem["initial_solutions"]
        types = np.asarray(
            [
                _node_type_value(t, "initial_solutions.types")
                for t in init["types"]
            ]
        )
        dm.add_initial_solutions(
            _i32(init["vehicle_ids"]),
            _i32(init["routes"]),
            types,
            _i32(init["sol_offsets"]),
        )

    return dm


def submit(problem: dict, settings: dict | None = None) -> dict:
    """Build a VRP model from plain JSON arrays and submit it for an
    asynchronous solve.

    Args:
        problem: The model as plain JSON arrays. See
            :func:`_build_routing_model_from_json` for the accepted keys --
            they mirror ``RoutingProblem`` in cuopt_routing.proto field for
            field (e.g. ``cost_matrices``, ``order_locations``,
            ``vehicle_time_windows``, ``capacity_dimensions``,
            ``pickup_delivery_pairs``, ``vehicle_breaks``, ``objective``).
        settings: Solver settings. Only ``time_limit``, ``verbose_mode``
            (or ``verbose``), and ``error_logging`` reach the server --
            ``dump_best_results``/``dump_config_file`` are local-solve-only
            and have no effect here.

    Returns
    -------
        A dict with ``job_id``, ``num_locations``, ``fleet_size``, and
        ``num_orders``.

    Raises
    ------
        CuOptMCPError: The problem is malformed, or the backend is
            unreachable.
    """
    model = _build_routing_model_from_json(problem)
    try:
        job_id = get_routing_client().submit(model, settings)
    except Exception as exc:
        raise describe_connection_error(exc) from exc
    return {
        "job_id": job_id,
        "num_locations": model.get_num_locations(),
        "fleet_size": model.get_fleet_size(),
        "num_orders": model.get_num_orders(),
        "next": (
            "Poll cuopt_status(job_id). When it reports COMPLETED, call "
            "cuopt_vrp_result(job_id)."
        ),
    }


def result(job_id: str, limit: int = INLINE_SOLUTION_LIMIT) -> dict:
    """Fetch a completed VRP solution, shaped to stay within a usable size.

    Args:
        job_id: A job handle previously returned by :func:`submit`.
        limit: Maximum route stops returned inline; must be between 0 and
            :data:`tools.INLINE_SOLUTION_LIMIT`. Beyond this, the full
            route table is written to a file and its path returned
            instead.

    Returns
    -------
        ``{"ready": False, ...}`` if the job hasn't finished yet. Otherwise
        a dict with ``status`` ("SUCCESS"/"FAIL"/"TIMEOUT"/"EMPTY"),
        ``total_objective_value``, ``objective_values`` (by name),
        ``vehicle_count``, and either ``stops`` (a flat list of
        ``{vehicle, location, type, arrival}``, one per visit across all
        routes) or, past ``limit``, ``stops_truncated`` plus
        ``solution_path``. A non-SUCCESS status carries
        ``status_message``/``error_message`` and, on FAIL,
        ``unserviced_orders``.

    Raises
    ------
        CuOptMCPError: ``limit`` is out of range, or the backend is
            unreachable.
    """
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 0 <= limit <= INLINE_SOLUTION_LIMIT
    ):
        raise CuOptMCPError(
            f"limit must be an integer between 0 and {INLINE_SOLUTION_LIMIT}"
        )
    try:
        sol = get_routing_client().result(job_id)
    except Exception as exc:
        raise describe_connection_error(exc) from exc
    if sol is None:
        return {
            "job_id": job_id,
            "ready": False,
            "hint": "Job has not finished. Poll cuopt_status(job_id).",
        }

    summary = {
        "job_id": job_id,
        "ready": True,
        "status": _STATUS_NAMES.get(sol["status"], str(sol["status"])),
        "vehicle_count": int(sol["vehicle_count"]),
        "total_objective_value": float(sol["total_objective_value"]),
        "objective_values": {
            _OBJECTIVE_NAMES.get(k, str(k)): float(v)
            for k, v in sol["objective_values"].items()
        },
    }
    if sol["status_message"]:
        summary["status_message"] = sol["status_message"]
    if sol["error_message"]:
        summary["error_message"] = sol["error_message"]
    if sol["status"] == 1:  # FAIL
        summary["unserviced_orders"] = [
            int(v) for v in sol["unserviced_nodes"]
        ]

    stops = [
        {
            "vehicle": int(truck_id),
            "location": int(loc),
            "type": _NODE_TYPE_NAMES.get(int(ntype), str(int(ntype))),
            "arrival": float(arrival),
        }
        for truck_id, loc, ntype, arrival in zip(
            sol["truck_id"],
            sol["locations"],
            sol["node_types"],
            sol["arrival_stamp"],
        )
    ]
    if len(stops) <= limit:
        summary["stops"] = stops
    else:
        summary["stops_truncated"] = True
        summary["stops_shown"] = limit
        summary["stops"] = stops[:limit]
        path = _solution_file_path(job_id).with_suffix(".vrp.json")
        path.write_text(json.dumps(stops, indent=1))
        summary["solution_path"] = str(path)
        summary["hint"] = (
            f"{len(stops)} stops exceed the inline limit of {limit}. The "
            "full route table is at solution_path."
        )
    return summary
