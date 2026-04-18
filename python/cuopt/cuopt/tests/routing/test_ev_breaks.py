# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf

from cuopt import routing
from cuopt.routing import utils

filename = utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_data_model(n_vehicles=3):
    """Minimal DataModel for API-level tests (no solver needed)."""
    n = 6
    rows = [
        [0, 10, 20, 15, 25, 30],
        [10,  0, 15, 20, 30, 25],
        [20, 15,  0, 10, 20, 15],
        [15, 20, 10,  0, 15, 20],
        [25, 30, 20, 15,  0, 10],
        [30, 25, 15, 20, 10,  0],
    ]
    d = routing.DataModel(n, n_vehicles)
    d.add_cost_matrix(cudf.DataFrame(rows))
    return d


# ---------------------------------------------------------------------------
# API / data-model tests (no solve)
# ---------------------------------------------------------------------------

def test_ev_break_api_single_cycle_defaults():
    """Single cycle with min_range=0: distance window is [0, max_range]."""
    d = _small_data_model()
    d.add_ev_break(0, max_range=100.0, charge_duration=15)

    breaks = d.get_non_uniform_breaks()
    assert 0 in breaks
    assert len(breaks[0]) == 1
    assert breaks[0][0]["distance_min"] == 0.0
    assert breaks[0][0]["distance_max"] == 100.0
    assert breaks[0][0]["duration"] == 15


def test_ev_break_api_int_vehicle_id():
    """An int vehicle_id is equivalent to a single-element list."""
    d_int = _small_data_model()
    d_int.add_ev_break(0, max_range=100.0, charge_duration=15)

    d_list = _small_data_model()
    d_list.add_ev_break([0], max_range=100.0, charge_duration=15)

    b_int = d_int.get_non_uniform_breaks()
    b_list = d_list.get_non_uniform_breaks()
    assert b_int[0][0]["distance_min"] == b_list[0][0]["distance_min"]
    assert b_int[0][0]["distance_max"] == b_list[0][0]["distance_max"]


def test_ev_break_api_min_range():
    """min_range shifts the start of the first cycle's distance window."""
    d = _small_data_model()
    d.add_ev_break(0, max_range=100.0, charge_duration=10, min_range=30.0)

    breaks = d.get_non_uniform_breaks()
    assert breaks[0][0]["distance_min"] == 30.0
    assert breaks[0][0]["distance_max"] == 100.0


def test_ev_break_api_multi_cycle():
    """n_cycles creates successive non-overlapping distance windows per cycle."""
    max_range = 100.0
    min_range = 20.0
    n_cycles = 3

    d = _small_data_model()
    d.add_ev_break(
        0,
        max_range=max_range,
        charge_duration=15,
        min_range=min_range,
        n_cycles=n_cycles,
    )

    breaks = d.get_non_uniform_breaks()
    assert len(breaks[0]) == n_cycles
    for k in range(n_cycles):
        assert breaks[0][k]["distance_min"] == k * max_range + min_range
        assert breaks[0][k]["distance_max"] == (k + 1) * max_range


def test_ev_break_api_multiple_vehicles():
    """A list of vehicle_ids applies breaks to every vehicle in the list."""
    d = _small_data_model(n_vehicles=4)
    vehicle_ids = [0, 1, 2]

    d.add_ev_break(vehicle_ids, max_range=100.0, charge_duration=15)

    breaks = d.get_non_uniform_breaks()
    for vid in vehicle_ids:
        assert vid in breaks
        assert len(breaks[vid]) == 1

    assert 3 not in breaks


def test_ev_break_api_charging_stations_stored():
    """Specified charging_stations are stored in the non_uniform_breaks dict."""
    d = _small_data_model()
    stations = cudf.Series([1, 2, 3], dtype="int32")

    d.add_ev_break(0, max_range=100.0, charge_duration=15, charging_stations=stations)

    breaks = d.get_non_uniform_breaks()
    stored_locs = breaks[0][0]["locations"].to_arrow().to_pylist()
    assert stored_locs == [1, 2, 3]


def test_ev_break_api_stacked_calls():
    """Two separate add_ev_break calls on the same vehicle accumulate breaks."""
    d = _small_data_model()
    d.add_ev_break(0, max_range=100.0, charge_duration=15)
    d.add_ev_break(0, max_range=100.0, charge_duration=15)

    breaks = d.get_non_uniform_breaks()
    assert len(breaks[0]) == 2


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_ev_break_invalid_vehicle_id():
    """Out-of-range vehicle_id raises an exception."""
    d = _small_data_model(n_vehicles=2)

    with pytest.raises(Exception):
        d.add_ev_break(5, max_range=100.0, charge_duration=15)

    with pytest.raises(Exception):
        d.add_ev_break(-1, max_range=100.0, charge_duration=15)


# ---------------------------------------------------------------------------
# Solver integration tests
# ---------------------------------------------------------------------------

def test_ev_break_solves():
    """Basic EV break solve: status 0 and at least one Break in the route."""
    vehicle_num = 20
    run_nodes = 100
    d = utils.create_data_model(
        filename, run_nodes=run_nodes, num_vehicles=vehicle_num
    )

    d.add_ev_break(
        vehicle_ids=list(range(vehicle_num)),
        max_range=200.0,
        charge_duration=15,
    )

    s = routing.SolverSettings()
    s.set_time_limit(30)
    sol = routing.Solve(d, s)

    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    assert (routes["type"] == "Break").any()


def test_ev_break_charging_stations_in_solution():
    """Break locations in the solution are a subset of charging_stations."""
    vehicle_num = 20
    run_nodes = 100
    d = utils.create_data_model(
        filename, run_nodes=run_nodes, num_vehicles=vehicle_num
    )

    station_ids = list(range(1, 30))
    charging_stations = cudf.Series(station_ids, dtype="int32")

    d.add_ev_break(
        vehicle_ids=list(range(vehicle_num)),
        max_range=200.0,
        charge_duration=15,
        charging_stations=charging_stations,
    )

    s = routing.SolverSettings()
    s.set_time_limit(30)
    sol = routing.Solve(d, s)

    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    for i in range(routes.shape[0]):
        if routes["type"][i] == "Break":
            assert routes["location"][i] in station_ids


def test_ev_break_multi_cycle():
    """n_cycles=2 produces breaks with both cycle indices (0 and 1) in routes."""
    vehicle_num = 10
    run_nodes = 100
    n_cycles = 2
    d = utils.create_data_model(
        filename, run_nodes=run_nodes, num_vehicles=vehicle_num
    )

    d.add_ev_break(
        vehicle_ids=list(range(vehicle_num)),
        max_range=200.0,
        charge_duration=15,
        n_cycles=n_cycles,
    )

    s = routing.SolverSettings()
    s.set_time_limit(30)
    sol = routing.Solve(d, s)

    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    break_dims_seen = set(
        routes.loc[routes["type"] == "Break", "route"].tolist()
    )
    # Both cycle 0 and cycle 1 breaks should appear in at least some routes
    assert 0 in break_dims_seen
    assert 1 in break_dims_seen

    # Per-vehicle break counts must equal n_cycles for every active vehicle
    break_counts = {}
    for i in range(routes.shape[0]):
        truck = routes["truck_id"][i]
        if routes["type"][i] == "Break":
            break_counts[truck] = break_counts.get(truck, 0) + 1

    for count in break_counts.values():
        assert count == n_cycles


def test_ev_break_mixed_fleet():
    """Only EV-designated vehicles have charging breaks in the solution."""
    vehicle_num = 20
    run_nodes = 100
    ev_count = vehicle_num // 2
    d = utils.create_data_model(
        filename, run_nodes=run_nodes, num_vehicles=vehicle_num
    )

    ev_ids = set(range(ev_count))
    d.add_ev_break(
        vehicle_ids=list(ev_ids),
        max_range=200.0,
        charge_duration=15,
    )

    s = routing.SolverSettings()
    s.set_time_limit(30)
    sol = routing.Solve(d, s)

    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    for i in range(routes.shape[0]):
        if routes["type"][i] == "Break":
            assert routes["truck_id"][i] in ev_ids
