# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a recorded routing problem (the store-then-build IR) to host arrays.

Walks ``DataModel._calls`` and produces plain host (numpy) arrays keyed by the
gRPC ``RoutingProblem`` field names, exporting any device (cuDF/cupy) inputs to
host at this point -- the "export on serialize" step of the mixed IR. The result
is proto-agnostic (a dict of host arrays) so a protobuf/gRPC layer can map it
onto the wire without this module depending on generated stubs.

Most setters map 1:1 -- ``set_<field>(array)`` writes host field ``<field>`` --
and are handled automatically, so they need no entry here. Only setters that
rename fields, take several arguments, or build nested/keyed structures get an
explicit handler in ``_HANDLERS``. A recorded ``add_*`` setter with no handler
raises, and ``test_serialize`` checks every recorded setter is exportable, so a
new unmapped setter fails loudly instead of being silently dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from cuopt.routing.vehicle_routing import DataModel


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


# --- handlers for setters that do NOT map set_<field>(array) -> field <field> ---


def _matrix(key):
    def handle(p, args):
        mat = _to_host(args[0]).astype(np.float32, copy=False)
        vehicle_type = int(args[1]) if len(args) > 1 else 0
        p[key].append(
            {"vehicle_type": vehicle_type, "values": mat.ravel(order="C")}
        )

    return handle


def _pair(key0, key1):
    def handle(p, args):
        p[key0] = _to_host(args[0])
        p[key1] = _to_host(args[1])

    return handle


def _match(key):
    def handle(p, args):
        p[key].append({"id": int(args[0]), "matches": _to_host(args[1])})

    return handle


def _scalar(key):
    def handle(p, args):
        p[key] = int(args[0])

    return handle


def _capacity(p, args):
    p["capacity_dimensions"].append(
        {
            "name": args[0],
            "demand": _to_host(args[1]),
            "capacity": _to_host(args[2]),
        }
    )


def _service_times(p, args):
    vehicle_id = int(args[1]) if len(args) > 1 else -1
    p["order_service_times"].append(
        {"vehicle_id": vehicle_id, "service_times": _to_host(args[0])}
    )


def _precedence(p, args):
    p["order_precedence"].append(
        {"order_id": int(args[0]), "preceding_orders": _to_host(args[1])}
    )


def _objective(p, args):
    p["objective"] = {
        "objectives": _to_host(args[0]),
        "weights": _to_host(args[1]),
    }


def _initial_solutions(p, args):
    p["initial_solutions"] = {
        "vehicle_ids": _to_host(args[0]),
        "routes": _to_host(args[1]),
        "types": _to_host(args[2]),
        "sol_offsets": _to_host(args[3]),
    }


def _uniform_break(p, args):
    p["uniform_breaks"].append(
        {
            "earliest": _to_host(args[0]),
            "latest": _to_host(args[1]),
            "duration": _to_host(args[2]),
        }
    )


def _vehicle_break(p, args):
    vehicle_id = int(args[0])
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
        (e for e in p["vehicle_breaks"] if e["vehicle_id"] == vehicle_id), None
    )
    if entry is None:
        entry = {"vehicle_id": vehicle_id, "breaks": []}
        p["vehicle_breaks"].append(entry)
    entry["breaks"].append(brk)


_HANDLERS = {
    "add_cost_matrix": _matrix("cost_matrices"),
    "add_transit_time_matrix": _matrix("transit_time_matrices"),
    "set_order_time_windows": _pair("order_tw_earliest", "order_tw_latest"),
    "set_vehicle_time_windows": _pair(
        "vehicle_tw_earliest", "vehicle_tw_latest"
    ),
    "set_vehicle_locations": _pair(
        "vehicle_start_locations", "vehicle_return_locations"
    ),
    "set_pickup_delivery_pairs": _pair("pickup_indices", "delivery_indices"),
    "add_capacity_dimension": _capacity,
    "set_order_service_times": _service_times,
    "add_vehicle_order_match": _match("vehicle_order_match"),
    "add_order_vehicle_match": _match("order_vehicle_match"),
    "add_order_precedence": _precedence,
    "add_break_dimension": _uniform_break,
    "add_vehicle_break": _vehicle_break,
    "set_objective_function": _objective,
    "add_initial_solutions": _initial_solutions,
    "set_min_vehicles": _scalar("min_vehicles"),
}

# list-valued fields the handlers append to (pre-initialized so order of calls
# does not matter).
_LIST_FIELDS = (
    "cost_matrices",
    "transit_time_matrices",
    "capacity_dimensions",
    "order_service_times",
    "vehicle_order_match",
    "order_vehicle_match",
    "order_precedence",
    "uniform_breaks",
    "vehicle_breaks",
)


def to_host_problem(dm: DataModel) -> dict[str, Any]:
    """Export a recorded routing problem to host arrays keyed by proto fields.

    Parameters
    ----------
    dm : cuopt.routing.DataModel
        A routing data model whose setter calls have been recorded (the
        store-then-build IR). Device (cuDF/cupy) inputs are copied to host
        here; host inputs are already numpy in the IR.

    Returns
    -------
    dict
        Host (numpy) arrays and scalars keyed by the gRPC ``RoutingProblem``
        field names (e.g. ``cost_matrices``, ``order_locations``,
        ``capacity_dimensions``). List-valued fields are always present
        (possibly empty); other fields appear only when the corresponding
        setter was called.

    Raises
    ------
    KeyError
        If a recorded ``add_*`` setter, or a multi-argument ``set_*`` setter,
        has no export mapping. Add an explicit handler to ``_HANDLERS``.
    """
    n_loc, fleet, n_ord = dm._init_args
    p = {
        "num_locations": int(n_loc),
        "fleet_size": int(fleet),
        "num_orders": int(n_loc if n_ord == -1 else n_ord),
    }
    for key in _LIST_FIELDS:
        p[key] = []

    for name, args, _ in dm._calls:
        handler = _HANDLERS.get(name)
        if handler is not None:
            handler(p, args)
        elif name.startswith("set_"):
            # 1:1 single-array setter: set_<field>(array) -> field <field>.
            # A multi-argument set_* needs an explicit handler; refuse to
            # export it here rather than silently drop the extra arguments.
            if len(args) != 1:
                raise KeyError(
                    f"no 1:1 export mapping for recorded setter {name!r}; add"
                    " a handler to _serialize._HANDLERS"
                )
            p[name[len("set_") :]] = _to_host(args[0])
        else:
            raise KeyError(
                f"no export mapping for recorded setter {name!r}; add a handler"
                " to _serialize._HANDLERS"
            )
    return p
