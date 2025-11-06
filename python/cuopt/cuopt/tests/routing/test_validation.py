# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf

from cuopt import routing


# Validates cost matrix exceptions
def test_dist_mat():
    cost_matrix = cudf.DataFrame(
        [
            [0, 5.0, 5.0, 5.0],
            [5.0, 0, 5.0, 5.0],
            [5.0, 5.0, 0, 5.0],
            [5.0, -5.0, 5.0, 0],
        ]
    )
    with pytest.raises(Exception) as exc:
        dm = routing.DataModel(3, 3)
        dm.add_cost_matrix(cost_matrix)
    assert (
        str(exc.value)
        == "Number of locations doesn't match number of locations in matrix"
    )
    with pytest.raises(Exception) as exc:
        dm = routing.DataModel(cost_matrix.shape[0], 3)
        dm.add_cost_matrix(cost_matrix[:3])
    assert str(exc.value) == "cost matrix is expected to be a square matrix"
    with pytest.raises(Exception) as exc:
        dm = routing.DataModel(cost_matrix.shape[0], 3)
        dm.add_cost_matrix(cost_matrix)
    assert (
        str(exc.value)
        == "All values in cost matrix must be greater than or equal to zero"
    )


# Validates non_negative, size and earliest<latest exceptions
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
    with pytest.raises(Exception) as exc:
        dm.set_vehicle_time_windows(vehicle_earliest_size, vehicle_latest)
        assert (
            str(exc.value)
            == "earliest times size doesn't match number of vehicles"
        )
    with pytest.raises(Exception) as exc:
        dm.set_vehicle_time_windows(vehicle_earliest_neg, vehicle_latest)
        assert (
            str(exc.value)
            == "earliest times must be greater than or equal to zero"
        )
    with pytest.raises(Exception) as exc:
        dm.set_vehicle_time_windows(vehicle_earliest_greater, vehicle_latest)
        assert (
            str(exc.value)
            == "All earliest times must be lesser than latest times"
        )


# Validates value range exception
def test_range():
    cost_matrix = cudf.DataFrame([[0, 5.0, 5.0], [5.0, 0, 5.0], [5.0, 5.0, 0]])
    dm = routing.DataModel(cost_matrix.shape[0], 3, 5)
    dm.add_cost_matrix(cost_matrix)
    with pytest.raises(Exception) as exc:
        order_locations = cudf.Series([0, 1, 2, 4, 1])
        dm.set_order_locations(order_locations)
        assert (
            str(exc.value)
            == "All values in order locations must be less than or equal to 3"
        )
