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

    d.add_ev_break(
        0, max_range=100.0, charge_duration=15, charging_stations=stations
    )

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


@pytest.fixture
def model():
    return _small_data_model(n_vehicles=3)


# fleet_size=3 → validate_range(vid, "vehicle id", 0, 3): fails for vid < 0 or vid > 3
@pytest.mark.parametrize("vid", [-1, 4, 100])
def test_ev_break_invalid_vehicle_id(model, vid):
    """Out-of-range vehicle id raises ValueError."""
    with pytest.raises(ValueError, match="vehicle id"):
        model.add_ev_break(vid, max_range=100.0, charge_duration=15)


@pytest.mark.parametrize("max_range", [0, -1, -100.0])
def test_max_range_must_be_positive(model, max_range):
    """max_range must be strictly positive."""
    with pytest.raises(ValueError, match="max range"):
        model.add_ev_break(0, max_range=max_range, charge_duration=10)


def test_negative_min_range_rejected(model):
    """min_range must be non-negative."""
    with pytest.raises(ValueError, match="min range"):
        model.add_ev_break(
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
        model.add_ev_break(
            0, max_range=max_range, charge_duration=10, min_range=min_range
        )


def test_negative_charge_duration_rejected(model):
    """charge_duration must be non-negative."""
    with pytest.raises(ValueError, match="charge duration"):
        model.add_ev_break(0, max_range=100.0, charge_duration=-1)


@pytest.mark.parametrize("n_cycles", [0, -1, -5])
def test_invalid_n_cycles_non_positive(model, n_cycles):
    """n_cycles <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="n_cycles"):
        model.add_ev_break(
            0, max_range=100.0, charge_duration=10, n_cycles=n_cycles
        )


@pytest.mark.parametrize("n_cycles", [1.5, "3"])
def test_invalid_n_cycles_wrong_type(model, n_cycles):
    """Non-integer n_cycles raises ValueError."""
    with pytest.raises(ValueError, match="n_cycles"):
        model.add_ev_break(
            0, max_range=100.0, charge_duration=10, n_cycles=n_cycles
        )


def test_charging_stations_out_of_range(model):
    """Charging station indices must be within [0, num_locations)."""
    bad_stations = cudf.Series([999], dtype="int32")
    with pytest.raises(ValueError, match="charging stations"):
        model.add_ev_break(
            0,
            max_range=100.0,
            charge_duration=10,
            charging_stations=bad_stations,
        )
