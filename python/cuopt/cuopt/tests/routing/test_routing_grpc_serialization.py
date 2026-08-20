# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage + unit tests for the VRP gRPC client serialization (no server).

These exercise the exact ``_populate`` path used by ``RoutingClient.submit`` via
the ``problem_summary`` probe, so a setter that is added to the DataModel but not
mapped by the client fails loudly here.
"""

import numpy as np
import pytest

from cuopt import routing
from cuopt.routing import _deferred

grpc_client = pytest.importorskip("cuopt.grpc.routing.grpc_client")
HANDLED_SETTERS = grpc_client.HANDLED_SETTERS
problem_summary = grpc_client.problem_summary


def test_client_maps_every_recordable_setter():
    """Every setter the deferred DataModel can record must be mapped by the
    gRPC client. Fails loudly if a new setter is added without a mapping.
    """
    missing = set(_deferred._SETTERS) - HANDLED_SETTERS
    assert not missing, (
        f"gRPC client _populate does not map recorded setters {sorted(missing)}; "
        "add a case to grpc_client._populate and to HANDLED_SETTERS"
    )
    # and no stale names that are not real setters
    stale = HANDLED_SETTERS - set(_deferred._SETTERS)
    assert not stale, f"HANDLED_SETTERS lists non-setters {sorted(stale)}"


def test_populate_scalar_matrix_and_dimension_fields():
    dm = routing.DataModel(5, 2, 5)
    cost = np.ones((5, 5), dtype=np.float32)
    np.fill_diagonal(cost, 0)
    dm.add_cost_matrix(cost, 0)
    dm.add_cost_matrix(cost * 2, 1)
    dm.add_transit_time_matrix(cost, 0)
    dm.set_vehicle_time_windows(
        np.zeros(2, np.int32), np.full(2, 100, np.int32)
    )
    dm.set_order_time_windows(np.zeros(5, np.int32), np.full(5, 50, np.int32))
    dm.set_order_locations(np.arange(5, dtype=np.int32))
    dm.set_order_prizes(np.ones(5, np.float32))
    dm.set_order_service_times(np.ones(5, np.int32))
    dm.add_capacity_dimension(
        "w", np.ones(5, np.int32), np.full(2, 10, np.int32)
    )
    dm.set_vehicle_types(np.array([0, 1], np.uint8))
    dm.set_vehicle_max_costs(np.full(2, 99.0, np.float32))
    dm.set_vehicle_max_times(np.full(2, 99.0, np.float32))
    dm.set_objective_function(
        np.array([0], np.int32), np.array([1.0], np.float32)
    )
    dm.set_min_vehicles(1)

    s = problem_summary(dm)
    assert (s["num_locations"], s["fleet_size"], s["num_orders"]) == (5, 2, 5)
    assert s["cost_matrices"] == 2
    assert s["transit_time_matrices"] == 1
    assert s["vehicle_tw_latest"] == 2
    assert s["order_tw_latest"] == 5
    assert s["order_locations"] == 5
    assert s["order_prizes"] == 5
    assert s["order_service_times"] == 1  # one (default) vehicle_id entry
    assert s["capacity_dimensions"] == 1
    assert s["vehicle_types"] == 2
    assert s["vehicle_max_costs"] == 2
    assert s["objectives"] == 1
    assert s["min_vehicles"] == 1


def test_populate_pickup_delivery():
    dm = routing.DataModel(5, 2, 4)
    cost = np.ones((5, 5), dtype=np.float32)
    np.fill_diagonal(cost, 0)
    dm.add_cost_matrix(cost)
    dm.set_pickup_delivery_pairs(
        np.array([1, 2], np.int32), np.array([3, 4], np.int32)
    )
    s = problem_summary(dm)
    assert s["pickup_indices"] == 2
    assert s["delivery_indices"] == 2


def test_populate_matches_and_precedence():
    dm = routing.DataModel(4, 2)
    cost = np.ones((4, 4), dtype=np.float32)
    np.fill_diagonal(cost, 0)
    dm.add_cost_matrix(cost)
    dm.add_vehicle_order_match(0, np.array([1, 2], np.int32))
    dm.add_order_vehicle_match(1, np.array([0], np.int32))
    dm.add_order_precedence(2, np.array([1], np.int32))
    s = problem_summary(dm)
    assert s["vehicle_order_match"] == 1
    assert s["order_vehicle_match"] == 1
    assert s["order_precedence"] == 1


def test_populate_breaks():
    dm = routing.DataModel(4, 2)
    cost = np.ones((4, 4), dtype=np.float32)
    np.fill_diagonal(cost, 0)
    dm.add_cost_matrix(cost)
    dm.add_break_dimension(
        np.full(2, 10, np.int32),
        np.full(2, 20, np.int32),
        np.full(2, 5, np.int32),
    )
    s = problem_summary(dm)
    assert s["uniform_breaks"] == 1


def test_populate_handles_pandas_host_inputs():
    """Pandas (host) inputs map identically to numpy.

    This is the natural input for a GPU-less client.
    """
    pd = pytest.importorskip("pandas")
    cost = np.ones((4, 4), dtype=np.float32)
    np.fill_diagonal(cost, 0)

    dm_np = routing.DataModel(4, 2)
    dm_np.add_cost_matrix(cost)
    dm_np.set_vehicle_time_windows(
        np.zeros(2, np.int32), np.full(2, 100, np.int32)
    )
    dm_np.add_capacity_dimension(
        "w", np.array([0, 1, 1, 1], np.int32), np.array([3, 3], np.int32)
    )

    dm_pd = routing.DataModel(4, 2)
    dm_pd.add_cost_matrix(pd.DataFrame(cost))
    dm_pd.set_vehicle_time_windows(pd.Series([0, 0]), pd.Series([100, 100]))
    dm_pd.add_capacity_dimension(
        "w", pd.Series([0, 1, 1, 1]), pd.Series([3, 3])
    )

    assert problem_summary(dm_np) == problem_summary(dm_pd)


def test_populate_handles_cudf_device_inputs():
    """Device (cuDF) inputs are copied to host by _populate, matching numpy."""
    cudf = pytest.importorskip("cudf")
    cost = np.ones((4, 4), dtype=np.float32)
    np.fill_diagonal(cost, 0)

    dm_np = routing.DataModel(4, 2)
    dm_np.add_cost_matrix(cost)
    dm_np.add_capacity_dimension(
        "w", np.array([0, 1, 1, 1], np.int32), np.array([3, 3], np.int32)
    )

    dm_cudf = routing.DataModel(4, 2)
    dm_cudf.add_cost_matrix(cost)
    dm_cudf.add_capacity_dimension(
        "w",
        cudf.Series([0, 1, 1, 1], dtype="int32"),
        cudf.Series([3, 3], dtype="int32"),
    )

    assert problem_summary(dm_np) == problem_summary(dm_cudf)


def test_unmapped_setter_raises():
    """A recorded setter with no mapping fails loudly (mirrors the fail-loud
    contract), rather than silently dropping data on the wire.
    """
    dm = routing.DataModel(3, 1)
    dm.add_cost_matrix(np.eye(3, dtype=np.float32))
    dm._calls.append(("set_made_up_field", (np.array([1, 2, 3]),), {}))
    with pytest.raises(KeyError, match="set_made_up_field"):
        problem_summary(dm)
