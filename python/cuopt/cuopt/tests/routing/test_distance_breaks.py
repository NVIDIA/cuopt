# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    d.add_distance_break(0, max_range=100.0, duration=15)

    breaks = d.get_non_uniform_breaks()
    assert 0 in breaks
    assert len(breaks[0]) == 1
    assert breaks[0][0]["distance_min"] == 0.0
    assert breaks[0][0]["distance_max"] == 100.0
    assert breaks[0][0]["duration"] == 15


def test_distance_break_api_int_vehicle_id():
    """An int vehicle_id is equivalent to a single-element list."""
    d_int = _small_data_model()
    d_int.add_distance_break(0, max_range=100.0, duration=15)

    d_list = _small_data_model()
    d_list.add_distance_break([0], max_range=100.0, duration=15)

    b_int = d_int.get_non_uniform_breaks()
    b_list = d_list.get_non_uniform_breaks()
    assert b_int[0][0]["distance_min"] == b_list[0][0]["distance_min"]
    assert b_int[0][0]["distance_max"] == b_list[0][0]["distance_max"]


def test_distance_break_api_min_range():
    """min_range sets the first cycle's soft cumulative-distance target."""
    d = _small_data_model()
    d.add_distance_break(0, max_range=100.0, duration=10, min_range=30.0)

    breaks = d.get_non_uniform_breaks()
    assert breaks[0][0]["distance_min"] == 30.0
    assert breaks[0][0]["distance_max"] == 100.0


def test_distance_break_api_multi_cycle():
    """n_cycles creates successive soft-target/hard-deadline pairs."""
    max_range = 100.0
    min_range = 20.0
    n_cycles = 3

    d = _small_data_model()
    d.add_distance_break(
        0,
        max_range=max_range,
        duration=15,
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

    d.add_distance_break(vehicle_ids, max_range=100.0, duration=15)

    breaks = d.get_non_uniform_breaks()
    for vid in vehicle_ids:
        assert vid in breaks
        assert len(breaks[vid]) == 1

    assert 3 not in breaks


def test_distance_break_api_locations_stored():
    """Specified locations are stored in the non_uniform_breaks dict."""
    d = _small_data_model()
    break_locs = cudf.Series([1, 2, 3], dtype="int32")

    d.add_distance_break(0, max_range=100.0, duration=15, locations=break_locs)

    breaks = d.get_non_uniform_breaks()
    stored_locs = breaks[0][0]["locations"].to_arrow().to_pylist()
    assert stored_locs == [1, 2, 3]


def test_distance_break_api_stacked_calls():
    """Two separate add_distance_break calls on the same vehicle accumulate breaks."""
    d = _small_data_model()
    d.add_distance_break(0, max_range=100.0, duration=15)
    d.add_distance_break(0, max_range=100.0, duration=15)

    breaks = d.get_non_uniform_breaks()
    assert len(breaks[0]) == 2


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    return _small_data_model(n_vehicles=3)


# fleet_size=3 → validate_range(vid, "vehicle id", 0, 3): fails for vid < 0 or vid > 2
@pytest.mark.parametrize("vid", [-1, 3, 100])
def test_distance_break_invalid_vehicle_id(model, vid):
    """Out-of-range vehicle id raises ValueError."""
    with pytest.raises(ValueError, match="vehicle id"):
        model.add_distance_break(vid, max_range=100.0, duration=15)


@pytest.mark.parametrize("max_range", [0, -1, -100.0])
def test_max_range_must_be_positive(model, max_range):
    """max_range must be strictly positive."""
    with pytest.raises(ValueError, match="max range"):
        model.add_distance_break(0, max_range=max_range, duration=10)


def test_negative_min_range_rejected(model):
    """min_range must be non-negative."""
    with pytest.raises(ValueError, match="min range"):
        model.add_distance_break(
            0, max_range=100.0, duration=10, min_range=-1.0
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
            0, max_range=max_range, duration=10, min_range=min_range
        )


def test_negative_duration_rejected(model):
    """Duration must be non-negative."""
    with pytest.raises(ValueError, match="duration"):
        model.add_distance_break(0, max_range=100.0, duration=-1)


@pytest.mark.parametrize("n_cycles", [0, -1, -5])
def test_invalid_n_cycles_non_positive(model, n_cycles):
    """n_cycles <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="n_cycles"):
        model.add_distance_break(
            0, max_range=100.0, duration=10, n_cycles=n_cycles
        )


@pytest.mark.parametrize("n_cycles", [1.5, "3"])
def test_invalid_n_cycles_wrong_type(model, n_cycles):
    """Non-integer n_cycles raises ValueError."""
    with pytest.raises(ValueError, match="n_cycles"):
        model.add_distance_break(
            0, max_range=100.0, duration=10, n_cycles=n_cycles
        )


def test_locations_out_of_range(model):
    """Break location indices must be within [0, num_locations)."""
    bad_locs = cudf.Series([999], dtype="int32")
    with pytest.raises(ValueError, match="break locations"):
        model.add_distance_break(
            0,
            max_range=100.0,
            duration=10,
            locations=bad_locs,
        )


# ---------------------------------------------------------------------------
# Solver tests (require GPU / actual solve)
# ---------------------------------------------------------------------------

_COST_3X3 = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

# depot=0, orders=1-2, break locations=3-4
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
    dm.add_distance_break(0, max_range=2.0, duration=1)
    dm.add_distance_break(1, max_range=2.0, duration=1)
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


@pytest.mark.parametrize("objective_mode", ["defaults", "omit", "disable"])
def test_default_distance_break_cost_weight(objective_mode):
    """Distance-break cost defaults to 1 when omitted; an explicit zero disables it."""
    dm = routing.DataModel(3, 1, 1)
    dm.add_cost_matrix(cudf.DataFrame(_COST_3X3, dtype="float32"))
    dm.set_order_locations(cudf.Series([1], dtype="int32"))
    dm.add_distance_break(
        0,
        max_range=100.0,
        duration=0,
        locations=cudf.Series([2], dtype="int32"),
        min_range=10.0,
    )
    if objective_mode != "defaults":
        objectives = [routing.Objective.COST]
        weights = [2.0 if objective_mode == "omit" else 1.0]
        if objective_mode == "disable":
            objectives.append(routing.Objective.DISTANCE_BREAK_COST)
            weights.append(0.0)
        dm.set_objective_function(
            cudf.Series(objectives),
            cudf.Series(weights, dtype="float32"),
        )

    sol = _solve(dm)
    assert sol.get_status() == 0
    objectives = sol.get_objective_values()
    assert objectives[routing.Objective.COST] == 3.0
    if objective_mode == "disable":
        assert routing.Objective.DISTANCE_BREAK_COST not in objectives
        assert sol.get_total_objective() == 3.0
    else:
        assert objectives[routing.Objective.DISTANCE_BREAK_COST] == 8.0
        assert sol.get_total_objective() == (
            14.0 if objective_mode == "omit" else 11.0
        )


def test_solve_break_at_break_location():
    """When break locations are specified, every break node lands at one of them."""
    order_locations = cudf.Series([1, 2], dtype="int32")
    locations = cudf.Series([3, 4], dtype="int32")

    dm = routing.DataModel(5, 2, 2)
    dm.add_cost_matrix(cudf.DataFrame(_COST_5X5, dtype="float32"))
    dm.set_order_locations(order_locations)
    dm.add_distance_break(
        0,
        max_range=2.0,
        duration=1,
        locations=locations,
    )
    dm.add_distance_break(
        1,
        max_range=2.0,
        duration=1,
        locations=locations,
    )
    dm.set_min_vehicles(2)

    sol = _solve(dm)
    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    break_loc_set = {3, 4}
    vehicles_with_breaks = set()
    for i in range(routes.shape[0]):
        if routes["type"][i] == "Break":
            vehicles_with_breaks.add(int(routes["truck_id"][i]))
            assert routes["location"][i] in break_loc_set
    assert vehicles_with_breaks == {0, 1}, (
        f"expected breaks on vehicles {{0, 1}}, got {vehicles_with_breaks}"
    )


def test_solve_multi_cycle_break_count():
    """Each used vehicle with 2 cycles receives exactly 2 break nodes."""
    order_locations = cudf.Series([1, 2], dtype="int32")

    dm = routing.DataModel(5, 2, 2)
    dm.add_cost_matrix(cudf.DataFrame(_COST_5X5, dtype="float32"))
    dm.set_order_locations(order_locations)
    for vid in [0, 1]:
        dm.add_distance_break(vid, max_range=2.0, duration=1)
        dm.add_distance_break(vid, max_range=4.0, duration=1, min_range=2.0)
    dm.set_min_vehicles(2)

    sol = _solve(dm)
    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    breaks_per_vehicle = {}
    for i in range(routes.shape[0]):
        if routes["type"][i] == "Break":
            vid = int(routes["truck_id"][i])
            breaks_per_vehicle[vid] = breaks_per_vehicle.get(vid, 0) + 1

    assert set(breaks_per_vehicle) == {0, 1}, (
        f"expected breaks on vehicles {{0, 1}}, got {set(breaks_per_vehicle)}"
    )
    for vid, cnt in breaks_per_vehicle.items():
        assert cnt == 2


def test_solve_break_distance_window_enforced():
    """Solver picks a longer route so the break lands inside [0, d_max=60]."""
    cost_asym = [
        [0, 100, 50],
        [100, 0, 5],
        [1, 55, 0],
    ]
    order_locations = cudf.Series([1], dtype="int32")
    locations = cudf.Series([2], dtype="int32")

    dm = routing.DataModel(3, 1, 1)
    dm.add_cost_matrix(cudf.DataFrame(cost_asym, dtype="float32"))
    dm.set_order_locations(order_locations)
    dm.add_distance_break(
        0,
        max_range=60.0,
        duration=0,
        locations=locations,
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

    Two cycle targets of 10 and 30, with hard limits of 20 and 40 and a high
    early-break penalty, make the solver prefer distinct break locations for
    each cycle on a 5-location unit-cost grid (arc 10 between distinct locations).
    """
    # depot(0), customers(1, 2), break locations(3, 4); arc 10 between distinct locations.
    cost = [[0 if i == j else 10 for j in range(5)] for i in range(5)]
    order_locations = cudf.Series([1, 2], dtype="int32")
    locations = cudf.Series([3, 4], dtype="int32")

    max_range = 20.0
    min_range = 10.0
    n_cycles = 2
    duration = 1

    dm = routing.DataModel(5, 2, 2)
    dm.add_cost_matrix(cudf.DataFrame(cost, dtype="float32"))
    dm.set_order_locations(order_locations)
    dm.add_distance_break(
        vehicle_ids=[0, 1],
        max_range=max_range,
        duration=duration,
        locations=locations,
        min_range=min_range,
        n_cycles=n_cycles,
    )
    dm.set_objective_function(
        cudf.Series(
            [routing.Objective.COST, routing.Objective.DISTANCE_BREAK_COST]
        ),
        cudf.Series([1.0, 100.0], dtype="float32"),
    )
    dm.set_min_vehicles(2)

    sol = _solve(dm)
    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    cost_flat = [c for row in cost for c in row]
    break_loc_set = {int(s) for s in locations.to_arrow().to_pylist()}

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
            assert loc in break_loc_set, (
                f"vehicle {vid} break at location {loc} not in break locations "
                f"{break_loc_set}"
            )

    assert set(breaks_per_vehicle) == {0, 1}, (
        f"expected breaks on vehicles {{0, 1}}, got {set(breaks_per_vehicle)}"
    )
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
    dm.add_distance_break(0, max_range=2.0, duration=1)
    dm.set_min_vehicles(2)

    sol = _solve(dm)
    assert sol.get_status() == 0

    routes = sol.get_route().to_pandas()
    found_break_v0 = False
    for i in range(routes.shape[0]):
        if routes["type"][i] == "Break":
            assert routes["truck_id"][i] == 0, (
                f"vehicle {routes['truck_id'][i]} should not have a distance break"
            )
            found_break_v0 = True
    assert found_break_v0, (
        "vehicle 0 has a distance break configured but received none"
    )
