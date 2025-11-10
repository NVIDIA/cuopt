# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf

from cuopt import routing


def test_empty_routes_with_breaks():
    cost_matrix = cudf.DataFrame(
        [
            [0.0, 1.0, 2.0, 2.0, 5.0, 9.0],
            [1.0, 0.0, 3.0, 3.0, 6.0, 10.0],
            [3.0, 4.0, 0.0, 3.0, 6.0, 10.0],
            [3.0, 4.0, 3.0, 0.0, 3.0, 7.0],
            [5.0, 6.0, 7.0, 7.0, 0.0, 4.0],
            [8.0, 9.0, 10.0, 10.0, 3.0, 0.0],
        ]
    )

    cost_matrix_1 = cudf.DataFrame(
        [
            [0.0, 2.0, 4.0, 4.0, 9.0, 14.0],
            [2.0, 0.0, 6.0, 6.0, 11.0, 16.0],
            [6.0, 8.0, 0.0, 4.0, 9.0, 14.0],
            [5.0, 7.0, 5.0, 0.0, 5.0, 10.0],
            [8.0, 10.0, 12.0, 12.0, 0.0, 5.0],
            [12.0, 14.0, 16.0, 16.0, 4.0, 0.0],
        ]
    )

    transit_time_matrix = cost_matrix.copy(deep=True)
    transit_time_matrix_1 = cost_matrix_1.copy(deep=True)

    vehcile_start = cudf.Series([0, 1, 0, 1, 0])

    vehicle_cap = cudf.Series([10, 12, 15, 8, 10])

    vehicle_eal = cudf.Series([0, 1, 3, 5, 20])

    vehicle_lat = cudf.Series([80, 40, 30, 80, 100])

    vehicle_break_eal = cudf.Series([20, 20, 20, 20, 20])

    vehicle_break_lat = cudf.Series([25, 25, 25, 25, 25])

    vehicle_duration = cudf.Series([1, 1, 1, 1, 1])

    task_locations = cudf.Series([1, 2, 3, 4, 5])

    demand = cudf.Series([3, 4, 4, 3, 2])

    task_time_eal = cudf.Series([3, 5, 1, 4, 0])

    task_time_latest = cudf.Series([20, 30, 20, 40, 30])

    task_serv = cudf.Series([3, 1, 8, 4, 0])

    veh_types = cudf.Series([1, 2, 1, 2, 1])

    dm = routing.DataModel(
        cost_matrix.shape[0], len(vehcile_start), len(task_locations)
    )

    dm.add_cost_matrix(cost_matrix, 1)
    dm.add_cost_matrix(cost_matrix_1, 2)

    dm.add_transit_time_matrix(transit_time_matrix, 1)
    dm.add_transit_time_matrix(transit_time_matrix_1, 2)

    dm.set_order_locations(task_locations)

    dm.set_vehicle_types(veh_types)

    dm.add_break_dimension(
        vehicle_break_eal, vehicle_break_lat, vehicle_duration
    )

    dm.add_capacity_dimension("1", demand, vehicle_cap)

    dm.add_vehicle_order_match(0, cudf.Series([0, 4]))

    dm.add_order_vehicle_match(0, cudf.Series([0]))
    dm.add_order_vehicle_match(4, cudf.Series([0]))

    dm.set_vehicle_time_windows(vehicle_eal, vehicle_lat)

    dm.set_order_time_windows(task_time_eal, task_time_latest)

    dm.set_order_service_times(task_serv)

    sol_set = routing.SolverSettings()

    sol = routing.Solve(dm, sol_set)

    assert sol.get_status() == 0

    solution_cudf = sol.get_route()
    for i, assign in enumerate(
        solution_cudf["truck_id"].unique().to_arrow().to_pylist()
    ):
        solution_vehicle_x = solution_cudf[solution_cudf["truck_id"] == assign]
        h_route = solution_vehicle_x["route"].to_arrow().to_pylist()
        route_len = len(h_route)
        assert route_len > 3
