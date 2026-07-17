# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf

from cuopt import routing
from cuopt.routing._deferred import _SETTERS
from cuopt.routing._serialize import _HANDLERS, to_host_problem


def test_every_setter_is_exportable():
    """Every recorded setter must be exportable -- either it has an explicit
    handler or it maps 1:1 (``set_<field>``). Fails loudly if a new setter is
    added without an export mapping, instead of silently dropping its data.
    """
    unmapped = [
        n for n in _SETTERS if n not in _HANDLERS and not n.startswith("set_")
    ]
    assert not unmapped, f"setters with no export mapping: {unmapped}"


COST = np.array(
    [
        [0, 4, 5, 2, 7],
        [3, 0, 6, 8, 1],
        [5, 2, 0, 4, 9],
        [6, 3, 7, 0, 2],
        [1, 8, 4, 5, 0],
    ],
    dtype=np.float32,
)


def _has_device_ref(o):
    if isinstance(o, dict):
        return any(_has_device_ref(v) for v in o.values())
    if isinstance(o, list):
        return any(_has_device_ref(v) for v in o)
    return hasattr(o, "__cuda_array_interface__")


def test_export_exports_host_and_device_to_host():
    d = routing.DataModel(5, 2)
    d.add_cost_matrix(COST)  # host (numpy)
    d.add_cost_matrix(cudf.DataFrame(COST + 1), 1)  # device (cuDF)
    d.set_order_time_windows(
        np.array([0, 0, 0, 0, 0], np.int32), np.array([9] * 5, np.int32)
    )
    d.set_order_prizes(cudf.Series([0, 2, 2, 2, 2]).astype("float32"))
    d.add_capacity_dimension(
        "demand",
        np.array([0, 1, 1, 1, 1], np.int32),
        np.array([10, 10], np.int32),
    )
    d.set_min_vehicles(1)

    p = to_host_problem(d)

    assert (p["num_locations"], p["fleet_size"], p["num_orders"]) == (5, 2, 5)
    # matrices export row-major float32 with the right vehicle_type; host and
    # device inputs both land on host.
    m0, m1 = p["cost_matrices"]
    assert m0["vehicle_type"] == 0 and m0["values"].dtype == np.float32
    np.testing.assert_array_equal(m0["values"], COST.ravel(order="C"))
    assert m1["vehicle_type"] == 1 and m1["values"].dtype == np.float32
    np.testing.assert_array_equal(m1["values"], (COST + 1).ravel(order="C"))
    # order time windows (both arrays) and prizes (device -> host)
    np.testing.assert_array_equal(
        p["order_tw_earliest"], np.zeros(5, np.int32)
    )
    np.testing.assert_array_equal(
        p["order_tw_latest"], np.full(5, 9, np.int32)
    )
    np.testing.assert_array_equal(
        p["order_prizes"], np.array([0, 2, 2, 2, 2], np.float32)
    )
    # capacity dimension: name and both arrays
    cap = p["capacity_dimensions"][0]
    assert cap["name"] == "demand"
    np.testing.assert_array_equal(
        cap["demand"], np.array([0, 1, 1, 1, 1], np.int32)
    )
    np.testing.assert_array_equal(
        cap["capacity"], np.array([10, 10], np.int32)
    )
    assert p["min_vehicles"] == 1
    # the exported problem holds no device references
    assert not _has_device_ref(p)


def test_export_breaks():
    d = routing.DataModel(4, 2)
    d.add_cost_matrix(np.zeros((4, 4), np.float32))
    d.add_break_dimension(
        np.array([10, 10], np.int32),
        np.array([20, 20], np.int32),
        np.array([5, 5], np.int32),
    )
    # two breaks for the same vehicle; second has device locations
    d.add_vehicle_break(0, 10, 20, 5, np.array([1, 2], np.int32))
    d.add_vehicle_break(0, 30, 40, 5, cudf.Series([3]).astype("int32"))

    p = to_host_problem(d)

    ub = p["uniform_breaks"][0]
    np.testing.assert_array_equal(ub["earliest"], np.array([10, 10], np.int32))
    np.testing.assert_array_equal(ub["latest"], np.array([20, 20], np.int32))
    np.testing.assert_array_equal(ub["duration"], np.array([5, 5], np.int32))

    veh0 = p["vehicle_breaks"][0]
    assert veh0["vehicle_id"] == 0 and len(veh0["breaks"]) == 2
    b0, b1 = veh0["breaks"]
    assert (b0["earliest"], b0["latest"], b0["duration"]) == (10, 20, 5)
    np.testing.assert_array_equal(b0["locations"], np.array([1, 2], np.int32))
    # second break's locations came in as cuDF -> exported to host
    assert (b1["earliest"], b1["latest"], b1["duration"]) == (30, 40, 5)
    np.testing.assert_array_equal(b1["locations"], np.array([3], np.int32))
    assert not _has_device_ref(p)


def test_multi_arg_set_without_handler_raises():
    """A set_* call that is not a single-array setter (and has no handler) must
    raise rather than silently drop its extra arguments.
    """
    fake = type(
        "FakeDM",
        (),
        {
            "_init_args": (1, 1, -1),
            "_calls": [("set_two_args", (np.zeros(1), np.zeros(1)), {})],
        },
    )()
    with pytest.raises(KeyError):
        to_host_problem(fake)
