# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf

from cuopt import routing

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_data_model(n_vehicles=3):
    """Minimal DataModel for API-level tests (no solver needed)."""
    n = 6
    rows = [
        [0, 10, 20, 15, 25, 30],
        [10, 0, 15, 20, 30, 25],
        [20, 15, 0, 10, 20, 15],
        [15, 20, 10, 0, 15, 20],
        [25, 30, 20, 15, 0, 10],
        [30, 25, 15, 20, 10, 0],
    ]
    d = routing.DataModel(n, n_vehicles)
    d.add_cost_matrix(cudf.DataFrame(rows, dtype="float32"))
    return d


# ---------------------------------------------------------------------------
# API / data-model tests (no solve)
# ---------------------------------------------------------------------------


def test_distance_break_api_single_cycle_defaults():
    """Single cycle with min_range=0: distance window is [0, max_range]."""
    d = _small_data_model()
    d.add_distance_break(0, max_range=100.0, charge_duration=15)

    breaks = d.get_non_uniform_breaks()
    assert 0 in breaks
    assert len(breaks[0]) == 1
    assert breaks[0][0]["distance_min"] == 0.0
    assert breaks[0][0]["distance_max"] == 100.0
    assert breaks[0][0]["duration"] == 15


def test_distance_break_api_int_vehicle_id():
    """An int vehicle_id is equivalent to a single-element list."""
    d_int = _small_data_model()
    d_int.add_distance_break(0, max_range=100.0, charge_duration=15)

    d_list = _small_data_model()
    d_list.add_distance_break([0], max_range=100.0, charge_duration=15)

    b_int = d_int.get_non_uniform_breaks()
    b_list = d_list.get_non_uniform_breaks()
    assert b_int[0][0]["distance_min"] == b_list[0][0]["distance_min"]
    assert b_int[0][0]["distance_max"] == b_list[0][0]["distance_max"]


def test_distance_break_api_min_range():
    """min_range shifts the start of the first cycle's distance window."""
    d = _small_data_model()
    d.add_distance_break(
        0, max_range=100.0, charge_duration=10, min_range=30.0
    )

    breaks = d.get_non_uniform_breaks()
    assert breaks[0][0]["distance_min"] == 30.0
    assert breaks[0][0]["distance_max"] == 100.0


def test_distance_break_api_multi_cycle():
    """n_cycles creates successive non-overlapping distance windows per cycle."""
    max_range = 100.0
    min_range = 20.0
    n_cycles = 3

    d = _small_data_model()
    d.add_distance_break(
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


def test_distance_break_api_multiple_vehicles():
    """A list of vehicle_ids applies breaks to every vehicle in the list."""
    d = _small_data_model(n_vehicles=4)
    vehicle_ids = [0, 1, 2]

    d.add_distance_break(vehicle_ids, max_range=100.0, charge_duration=15)

    breaks = d.get_non_uniform_breaks()
    for vid in vehicle_ids:
        assert vid in breaks
        assert len(breaks[vid]) == 1

    assert 3 not in breaks


def test_distance_break_api_charging_stations_stored():
    """Specified charging_stations are stored in the non_uniform_breaks dict."""
    d = _small_data_model()
    stations = cudf.Series([1, 2, 3], dtype="int32")

    d.add_distance_break(
        0, max_range=100.0, charge_duration=15, charging_stations=stations
    )

    breaks = d.get_non_uniform_breaks()
    stored_locs = breaks[0][0]["locations"].to_arrow().to_pylist()
    assert stored_locs == [1, 2, 3]


def test_distance_break_api_stacked_calls():
    """Two separate add_distance_break calls on the same vehicle accumulate breaks."""
    d = _small_data_model()
    d.add_distance_break(0, max_range=100.0, charge_duration=15)
    d.add_distance_break(0, max_range=100.0, charge_duration=15)

    breaks = d.get_non_uniform_breaks()
    assert len(breaks[0]) == 2


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    return _small_data_model(n_vehicles=3)


# fleet_size=3 → validate_range(vid, "vehicle id", 0, 3): fails for vid < 0 or vid > 3
@pytest.mark.parametrize("vid", [-1, 4, 100])
def test_distance_break_invalid_vehicle_id(model, vid):
    """Out-of-range vehicle id raises ValueError."""
    with pytest.raises(ValueError, match="vehicle id"):
        model.add_distance_break(vid, max_range=100.0, charge_duration=15)


@pytest.mark.parametrize("max_range", [0, -1, -100.0])
def test_max_range_must_be_positive(model, max_range):
    """max_range must be strictly positive."""
    with pytest.raises(ValueError, match="max range"):
        model.add_distance_break(0, max_range=max_range, charge_duration=10)


def test_negative_min_range_rejected(model):
    """min_range must be non-negative."""
    with pytest.raises(ValueError, match="min range"):
        model.add_distance_break(
            0, max_range=100.0, charge_duration=10, min_range=-1.0
        )


@pytest.mark.parametrize(
    "min_range, max_range",
    [
        (100.0, 100.0),
        (150.0, 100.0),
    ],
)
def test_min_range_must_be_less_than_max_range(model, min_range, max_range):
    """min_range >= max_range raises ValueError."""
    with pytest.raises(ValueError, match="min_range must be smaller"):
        model.add_distance_break(
            0, max_range=max_range, charge_duration=10, min_range=min_range
        )


def test_negative_charge_duration_rejected(model):
    """charge_duration must be non-negative."""
    with pytest.raises(ValueError, match="charge duration"):
        model.add_distance_break(0, max_range=100.0, charge_duration=-1)


@pytest.mark.parametrize("n_cycles", [0, -1, -5])
def test_invalid_n_cycles_non_positive(model, n_cycles):
    """n_cycles <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="n_cycles"):
        model.add_distance_break(
            0, max_range=100.0, charge_duration=10, n_cycles=n_cycles
        )


@pytest.mark.parametrize("n_cycles", [1.5, "3"])
def test_invalid_n_cycles_wrong_type(model, n_cycles):
    """Non-integer n_cycles raises ValueError."""
    with pytest.raises(ValueError, match="n_cycles"):
        model.add_distance_break(
            0, max_range=100.0, charge_duration=10, n_cycles=n_cycles
        )


def test_charging_stations_out_of_range(model):
    """Charging station indices must be within [0, num_locations)."""
    bad_stations = cudf.Series([999], dtype="int32")
    with pytest.raises(ValueError, match="charging stations"):
        model.add_distance_break(
            0,
            max_range=100.0,
            charge_duration=10,
            charging_stations=bad_stations,
        )


# ---------------------------------------------------------------------------
# Solver tests (require GPU / actual solve)
# ---------------------------------------------------------------------------

_COST_3X3 = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

# depot=0, orders=1-2, charging stations=3-4
_COST_5X5 = [
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0],
]


def _solve(dm, time_limit=10):
    s = routing.SolverSettings()
    s.set_time_limit(time_limit)
    return routing.Solve(dm, s)


def test_solve_basic_break_assigned():
    """Each vehicle with a distance break receives exactly one break in the solution."""
    dm = routing.DataModel(3, 2)
    dm.add_cost_matrix(cudf.DataFrame(_COST_3X3, dtype="float32"))
    dm.add_distance_break(0, max_range=2.0, charge_duration=1)
    dm.add_distance_break(1, max_range=2.0, charge_duration=1)
    dm.set_min_vehicles(2)

    sol = _solve(dm)
    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    breaks_per_vehicle = {}
    for i in range(routes.shape[0]):
        if routes["type"][i] == "Break":
            vid = routes["truck_id"][i]
            breaks_per_vehicle[vid] = breaks_per_vehicle.get(vid, 0) + 1

    assert 0 in breaks_per_vehicle
    assert 1 in breaks_per_vehicle
    assert breaks_per_vehicle[0] == 1
    assert breaks_per_vehicle[1] == 1


def test_solve_no_regression_without_distance_break():
    """Baseline: an identical problem with no distance break configured solves successfully and
    produces zero break nodes, confirming the dimension is invisible when its data is absent.
    """
    dm = routing.DataModel(3, 2)
    dm.add_cost_matrix(cudf.DataFrame(_COST_3X3, dtype="float32"))
    dm.set_min_vehicles(2)

    sol = _solve(dm)
    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    for i in range(routes.shape[0]):
        assert routes["type"][i] != "Break", (
            "no distance break configured but solver emitted a break node"
        )


def test_solve_break_at_charging_station():
    """When charging_stations are specified, every break node lands at one of them."""
    order_locations = cudf.Series([1, 2], dtype="int32")
    charging_stations = cudf.Series([3, 4], dtype="int32")

    dm = routing.DataModel(5, 2, 2)
    dm.add_cost_matrix(cudf.DataFrame(_COST_5X5, dtype="float32"))
    dm.set_order_locations(order_locations)
    dm.add_distance_break(
        0,
        max_range=2.0,
        charge_duration=1,
        charging_stations=charging_stations,
    )
    dm.add_distance_break(
        1,
        max_range=2.0,
        charge_duration=1,
        charging_stations=charging_stations,
    )
    dm.set_min_vehicles(2)

    sol = _solve(dm)
    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    station_set = {3, 4}
    for i in range(routes.shape[0]):
        if routes["type"][i] == "Break":
            assert routes["location"][i] in station_set


def test_solve_multi_cycle_break_count():
    """Each used vehicle with 2 cycles receives exactly 2 break nodes."""
    order_locations = cudf.Series([1, 2], dtype="int32")

    dm = routing.DataModel(5, 2, 2)
    dm.add_cost_matrix(cudf.DataFrame(_COST_5X5, dtype="float32"))
    dm.set_order_locations(order_locations)
    for vid in [0, 1]:
        dm.add_distance_break(vid, max_range=2.0, charge_duration=1)
        dm.add_distance_break(
            vid, max_range=4.0, charge_duration=1, min_range=2.0
        )
    dm.set_min_vehicles(2)

    sol = _solve(dm)
    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    breaks_per_vehicle = {}
    for i in range(routes.shape[0]):
        if routes["type"][i] == "Break":
            vid = routes["truck_id"][i]
            breaks_per_vehicle[vid] = breaks_per_vehicle.get(vid, 0) + 1

    for vid, cnt in breaks_per_vehicle.items():
        assert cnt == 2


def test_solve_break_distance_window_enforced():
    """Solver picks the more expensive charger-first ordering because the cheaper
    customer-first route would place the break past its d_max=60 window.
    """
    cost_asym = [
        [0, 100, 50],
        [100, 0, 5],
        [1, 55, 0],
    ]
    order_locations = cudf.Series([1], dtype="int32")
    charging_stations = cudf.Series([2], dtype="int32")

    dm = routing.DataModel(3, 1, 1)
    dm.add_cost_matrix(cudf.DataFrame(cost_asym, dtype="float32"))
    dm.set_order_locations(order_locations)
    dm.add_distance_break(
        0,
        max_range=60.0,
        charge_duration=0,
        charging_stations=charging_stations,
    )

    sol = _solve(dm)
    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    cost_flat = [c for row in cost_asym for c in row]
    cumulative = 0.0
    prev_loc = 0
    found_break = False
    for i in range(routes.shape[0]):
        loc = int(routes["location"][i])
        cumulative += cost_flat[prev_loc * 3 + loc]
        if routes["type"][i] == "Break":
            found_break = True
            assert cumulative <= 60.0, (
                f"break at cumulative distance {cumulative} exceeds d_max=60"
            )
        prev_loc = loc

    assert found_break, "no break found in solution"


def test_solve_full_feature_api():
    """Exercises every add_distance_break parameter at non-default values.

    Two non-overlapping cycle windows [10, 20] and [30, 40] separated by a 10-unit gap
    force the solver to pick distinct charging stations for each cycle on a 5-location
    unit-cost grid (arc 10 between any two distinct locations).
    """
    # depot(0), customers(1, 2), charging stations(3, 4); arc 10 between distinct locations.
    cost = [[0 if i == j else 10 for j in range(5)] for i in range(5)]
    order_locations = cudf.Series([1, 2], dtype="int32")
    charging_stations = cudf.Series([3, 4], dtype="int32")

    max_range = 20.0
    min_range = 10.0
    n_cycles = 2
    charge_duration = 1

    dm = routing.DataModel(5, 2, 2)
    dm.add_cost_matrix(cudf.DataFrame(cost, dtype="float32"))
    dm.set_order_locations(order_locations)
    dm.add_distance_break(
        vehicle_ids=[0, 1],
        max_range=max_range,
        charge_duration=charge_duration,
        charging_stations=charging_stations,
        min_range=min_range,
        n_cycles=n_cycles,
    )
    dm.set_min_vehicles(2)

    sol = _solve(dm)
    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    cost_flat = [c for row in cost for c in row]
    station_set = {int(s) for s in charging_stations.to_arrow().to_pylist()}

    cycle_windows = [
        (k * max_range + min_range, (k + 1) * max_range)
        for k in range(n_cycles)
    ]
    breaks_per_vehicle: dict[int, list[float]] = {}
    cumulative_per_vehicle: dict[int, float] = {}
    prev_loc_per_vehicle: dict[int, int] = {}
    n_loc = 5

    for i in range(routes.shape[0]):
        vid = int(routes["truck_id"][i])
        loc = int(routes["location"][i])
        prev_loc = prev_loc_per_vehicle.get(vid, 0)
        cumulative_per_vehicle[vid] = (
            cumulative_per_vehicle.get(vid, 0.0)
            + cost_flat[prev_loc * n_loc + loc]
        )
        prev_loc_per_vehicle[vid] = loc

        if routes["type"][i] == "Break":
            breaks_per_vehicle.setdefault(vid, []).append(
                cumulative_per_vehicle[vid]
            )
            assert loc in station_set, (
                f"vehicle {vid} break at location {loc} not in charging stations "
                f"{station_set}"
            )

    assert breaks_per_vehicle, "no vehicle received a break"
    for vid, break_distances in breaks_per_vehicle.items():
        assert len(break_distances) == n_cycles, (
            f"vehicle {vid} has {len(break_distances)} breaks, expected {n_cycles}"
        )
        for k, d in enumerate(break_distances):
            lo, hi = cycle_windows[k]
            assert lo - 1e-6 <= d <= hi + 1e-6, (
                f"vehicle {vid} cycle {k} break at cumulative {d} outside window "
                f"[{lo}, {hi}]"
            )


def test_solve_mixed_fleet_break_assignment():
    """Only the vehicle with a distance break configured receives break nodes."""
    dm = routing.DataModel(3, 2)
    dm.add_cost_matrix(cudf.DataFrame(_COST_3X3, dtype="float32"))
    dm.add_distance_break(0, max_range=2.0, charge_duration=1)
    dm.set_min_vehicles(2)

    sol = _solve(dm)
    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    for i in range(routes.shape[0]):
        if routes["type"][i] == "Break":
            assert routes["truck_id"][i] == 0, (
                f"vehicle {routes['truck_id'][i]} should not have a distance break"
            )
