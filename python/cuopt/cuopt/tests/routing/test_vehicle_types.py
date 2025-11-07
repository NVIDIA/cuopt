# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf

from cuopt import routing


def test_multiple_set_cost_matrix():
    with pytest.raises(ValueError):
        bikes_type = 1
        bikes_cost = cudf.DataFrame([[0, 4, 4], [4, 0, 4], [4, 4, 0]])
        bikes_time = cudf.DataFrame([[0, 50, 50], [50, 0, 50], [50, 50, 0]])

        dm = routing.DataModel(3, 2)
        dm.add_cost_matrix(bikes_cost, bikes_type)
        dm.add_transit_time_matrix(bikes_time, bikes_type)
        dm.add_cost_matrix(bikes_cost, bikes_type)


def test_multiple_set_time_matrix():
    with pytest.raises(ValueError):
        bikes_type = 1
        bikes_cost = cudf.DataFrame([[0, 4, 4], [4, 0, 4], [4, 4, 0]])
        bikes_time = cudf.DataFrame([[0, 50, 50], [50, 0, 50], [50, 50, 0]])

        dm = routing.DataModel(3, 2)
        dm.add_cost_matrix(bikes_cost, bikes_type)
        dm.add_transit_time_matrix(bikes_time, bikes_type)
        dm.add_transit_time_matrix(bikes_time, bikes_type)


def test_vehicle_types():
    bikes_type = 1
    car_type = 2

    bikes_cost = cudf.DataFrame([[0, 4, 4], [4, 0, 4], [4, 4, 0]])
    bikes_time = cudf.DataFrame([[0, 50, 50], [50, 0, 50], [50, 50, 0]])
    car_cost = cudf.DataFrame([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    car_time = cudf.DataFrame([[0, 10, 10], [10, 0, 10], [10, 10, 0]])
    vehicle_types = cudf.Series([bikes_type, car_type])

    dm = routing.DataModel(3, 2)
    dm.add_cost_matrix(bikes_cost, bikes_type)
    dm.add_transit_time_matrix(bikes_time, bikes_type)
    dm.add_cost_matrix(car_cost, car_type)
    dm.add_transit_time_matrix(car_time, car_type)
    dm.set_vehicle_types(vehicle_types)
    dm.set_min_vehicles(2)

    s = routing.SolverSettings()
    s.set_time_limit(1)

    sol = routing.Solve(dm, s)

    cost = sol.get_total_objective()
    cu_status = sol.get_status()
    vehicle_count = sol.get_vehicle_count()
    assert cu_status == 0
    assert vehicle_count == 2
    assert cost == 10
    solution_cudf = sol.get_route()

    for i, assign in enumerate(
        solution_cudf["truck_id"].unique().to_arrow().to_pylist()
    ):
        solution_vehicle_x = solution_cudf[solution_cudf["truck_id"] == assign]
        vehicle_x_start_time = round(
            float(solution_vehicle_x["arrival_stamp"].min()), 2
        )
        vehicle_x_final_time = round(
            float(solution_vehicle_x["arrival_stamp"].max()), 2
        )
        vehicle_x_total_time = round(
            vehicle_x_final_time - vehicle_x_start_time, 2
        )

        if vehicle_types[assign] == bikes_type:
            assert abs(vehicle_x_total_time - 100) < 0.01

        if vehicle_types[assign] == car_type:
            assert abs(vehicle_x_total_time - 20) < 0.01
