# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest

import cudf

from cuopt import routing
from cuopt.utilities import InputValidationError


# ----- Warnings -----


def test_type_casting_warnings():
    cost_matrix = cudf.DataFrame([[0, 4, 4], [4, 0, 4], [4, 4, 0]])
    constraints = cudf.DataFrame()
    constraints["earliest"] = [0, 0, 0]
    constraints["latest"] = [45, 45, 45]
    constraints["service"] = [2.5, 2.5, 2.5]

    dm = routing.DataModel(3, 2)
    with warnings.catch_warnings(record=True) as w:
        dm.add_cost_matrix(cost_matrix)
        assert "Casting cost_matrix from int64 to float32" in str(w[0].message)

        dm.set_order_time_windows(
            constraints["earliest"], constraints["latest"]
        )

        dm.set_order_service_times(constraints["service"])
        assert "Casting service_times from float64 to int32" in str(
            w[1].message
        )


# ----- Validation (matrix, time windows, range) -----


def test_dist_mat():
    cost_matrix = cudf.DataFrame(
        [
            [0, 5.0, 5.0, 5.0],
            [5.0, 0, 5.0, 5.0],
            [5.0, 5.0, 0, 5.0],
            [5.0, -5.0, 5.0, 0],
        ]
    )
    with pytest.raises(Exception) as exc_info:
        dm = routing.DataModel(3, 3)
        dm.add_cost_matrix(cost_matrix)
    assert (
        str(exc_info.value)
        == "Number of locations doesn't match number of locations in matrix"
    )
    with pytest.raises(Exception) as exc_info:
        dm = routing.DataModel(cost_matrix.shape[0], 3)
        dm.add_cost_matrix(cost_matrix[:3])
    assert (
        str(exc_info.value) == "cost matrix is expected to be a square matrix"
    )
    with pytest.raises(Exception) as exc_info:
        dm = routing.DataModel(cost_matrix.shape[0], 3)
        dm.add_cost_matrix(cost_matrix)
    assert (
        str(exc_info.value)
        == "All values in cost matrix must be greater than or equal to zero"
    )


def test_time_windows():
    cost_matrix = cudf.DataFrame(
        [
            [0, 5.0, 5.0, 5.0],
            [5.0, 0, 5.0, 5.0],
            [5.0, 5.0, 0, 5.0],
            [5.0, 5.0, 5.0, 0],
        ]
    )
    dm = routing.DataModel(cost_matrix.shape[0], 3)
    dm.add_cost_matrix(cost_matrix)

    vehicle_start = cudf.Series([1, 2, 3])
    vehicle_return = cudf.Series([1, 2, 3])
    vehicle_earliest_size = cudf.Series([60, 60])
    vehicle_earliest_neg = cudf.Series([-60, 60, 60])
    vehicle_earliest_greater = cudf.Series([60, 60, 120])
    vehicle_latest = cudf.Series([100] * 3)

    dm.set_vehicle_locations(vehicle_start, vehicle_return)
    with pytest.raises(Exception) as exc_info:
        dm.set_vehicle_time_windows(vehicle_earliest_size, vehicle_latest)
    assert (
        str(exc_info.value)
        == "earliest times size doesn't match number of vehicles"
    )
    with pytest.raises(Exception) as exc_info:
        dm.set_vehicle_time_windows(vehicle_earliest_neg, vehicle_latest)
    assert (
        str(exc_info.value)
        == "All values in earliest times  must be greater than or equal to zero"
    )
    with pytest.raises(Exception) as exc_info:
        dm.set_vehicle_time_windows(vehicle_earliest_greater, vehicle_latest)
    assert (
        str(exc_info.value)
        == "All earliest times must be lesser than latest times"
    )


def test_range():
    cost_matrix = cudf.DataFrame([[0, 5.0, 5.0], [5.0, 0, 5.0], [5.0, 5.0, 0]])
    dm = routing.DataModel(cost_matrix.shape[0], 3, 5)
    dm.add_cost_matrix(cost_matrix)
    with pytest.raises(Exception) as exc_info:
        order_locations = cudf.Series([0, 1, 2, 4, 1])
        dm.set_order_locations(order_locations)
    assert (
        str(exc_info.value)
        == "All values in order locations must be less than or equal to 3"
    )


def test_invalid_datamodel():
    with pytest.raises(InputValidationError) as err:
        routing.DataModel(0, 0, 0)
    assert str(err.value) == "The data model needs at least one location"
