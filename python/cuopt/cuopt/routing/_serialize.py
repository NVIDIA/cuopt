# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a recorded routing problem (the store-then-build IR) to host arrays.

Walks ``DataModel._calls`` and produces plain host (numpy) arrays keyed by the
gRPC ``RoutingProblem`` field names, exporting any device (cuDF/cupy) inputs to
host at this point -- the "export on serialize" step of the mixed IR. The result
is proto-agnostic (a dict of host arrays) so a protobuf/gRPC layer can map it
onto the wire without this module depending on generated stubs.
"""

import numpy as np


def _to_host(x):
    """Return a host numpy view/copy of ``x`` (numpy/pandas/cuDF/cupy/list)."""
    if isinstance(x, np.ndarray):
        return x
    root = type(x).__module__.split(".", 1)[0]
    if root in ("pandas", "cudf"):
        return x.to_numpy()
    if root == "cupy":
        return x.get()
    return np.asarray(x)


def _matrix(bucket, args):
    mat = _to_host(args[0]).astype(np.float32, copy=False)
    vehicle_type = int(args[1]) if len(args) > 1 else 0
    bucket.append(
        {"vehicle_type": vehicle_type, "values": mat.ravel(order="C")}
    )


def to_host_problem(dm):
    """Export ``dm``'s recorded problem as host arrays keyed by RoutingProblem
    fields. Device (cuDF/cupy) inputs are copied to host here; host inputs are
    already numpy in the IR.
    """
    n_loc, fleet, n_ord = dm._init_args
    p = {
        "num_locations": int(n_loc),
        "fleet_size": int(fleet),
        "num_orders": int(n_loc if n_ord == -1 else n_ord),
        "cost_matrices": [],
        "transit_time_matrices": [],
        "capacity_dimensions": [],
        "order_service_times": [],
        "vehicle_order_match": [],
        "order_vehicle_match": [],
        "order_precedence": [],
        "uniform_breaks": [],
        "vehicle_breaks": [],
    }

    def put(key, value):
        p[key] = _to_host(value)

    for name, args, _ in dm._calls:
        if name == "add_cost_matrix":
            _matrix(p["cost_matrices"], args)
        elif name == "add_transit_time_matrix":
            _matrix(p["transit_time_matrices"], args)
        elif name == "set_vehicle_locations":
            put("vehicle_start_locations", args[0])
            put("vehicle_return_locations", args[1])
        elif name == "set_vehicle_time_windows":
            put("vehicle_tw_earliest", args[0])
            put("vehicle_tw_latest", args[1])
        elif name == "set_vehicle_types":
            put("vehicle_types", args[0])
        elif name == "set_drop_return_trips":
            put("drop_return_trips", args[0])
        elif name == "set_skip_first_trips":
            put("skip_first_trips", args[0])
        elif name == "set_vehicle_max_costs":
            put("vehicle_max_costs", args[0])
        elif name == "set_vehicle_max_times":
            put("vehicle_max_times", args[0])
        elif name == "set_vehicle_fixed_costs":
            put("vehicle_fixed_costs", args[0])
        elif name == "set_order_locations":
            put("order_locations", args[0])
        elif name == "set_order_time_windows":
            put("order_tw_earliest", args[0])
            put("order_tw_latest", args[1])
        elif name == "set_order_prizes":
            put("order_prizes", args[0])
        elif name == "set_order_service_times":
            vid = int(args[1]) if len(args) > 1 else -1
            p["order_service_times"].append(
                {"vehicle_id": vid, "service_times": _to_host(args[0])}
            )
        elif name == "set_pickup_delivery_pairs":
            put("pickup_indices", args[0])
            put("delivery_indices", args[1])
        elif name == "add_capacity_dimension":
            p["capacity_dimensions"].append(
                {
                    "name": args[0],
                    "demand": _to_host(args[1]),
                    "capacity": _to_host(args[2]),
                }
            )
        elif name == "set_objective_function":
            p["objective"] = {
                "objectives": _to_host(args[0]),
                "weights": _to_host(args[1]),
            }
        elif name == "set_min_vehicles":
            p["min_vehicles"] = int(args[0])
        elif name == "add_vehicle_order_match":
            p["vehicle_order_match"].append(
                {"id": int(args[0]), "matches": _to_host(args[1])}
            )
        elif name == "add_order_vehicle_match":
            p["order_vehicle_match"].append(
                {"id": int(args[0]), "matches": _to_host(args[1])}
            )
        elif name == "add_order_precedence":
            p["order_precedence"].append(
                {
                    "order_id": int(args[0]),
                    "preceding_orders": _to_host(args[1]),
                }
            )
        elif name == "set_break_locations":
            put("break_locations", args[0])
        elif name == "add_initial_solutions":
            p["initial_solutions"] = {
                "vehicle_ids": _to_host(args[0]),
                "routes": _to_host(args[1]),
                "types": _to_host(args[2]),
                "sol_offsets": _to_host(args[3]),
            }
        elif name == "add_break_dimension":
            p["uniform_breaks"].append(
                {
                    "earliest": _to_host(args[0]),
                    "latest": _to_host(args[1]),
                    "duration": _to_host(args[2]),
                }
            )
        elif name == "add_vehicle_break":
            vid = int(args[0])
            locations = args[4] if len(args) > 4 else None
            brk = {
                "earliest": int(args[1]),
                "latest": int(args[2]),
                "duration": int(args[3]),
                "locations": (
                    _to_host(locations)
                    if locations is not None
                    else np.empty(0, np.int32)
                ),
            }
            entry = next(
                (e for e in p["vehicle_breaks"] if e["vehicle_id"] == vid),
                None,
            )
            if entry is None:
                entry = {"vehicle_id": vid, "breaks": []}
                p["vehicle_breaks"].append(entry)
            entry["breaks"].append(brk)
    return p
