# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf

from cuopt import routing


def check_cuopt_solution(
    routing_solution,
    distance_matrix,
    time_matrix,
    earliest_time,
    latest_time,
    v_service_times,
):
    th = 0.001
    df_distance_matrix = distance_matrix.to_pandas().values
    df_time_matrix = time_matrix.to_pandas().values
    df_earliest_time = earliest_time.to_pandas().values
    df_latest_time = latest_time.to_pandas().values
    routes = routing_solution.get_route()
    computed_cost = 0

    for truck_id, assign in enumerate(
        routes["truck_id"].unique().to_arrow().to_pylist()
    ):
        solution_vehicle_x = routes[routes["truck_id"] == assign]
        vehicle_x_total_time = float(solution_vehicle_x["arrival_stamp"].max())
        arrival_time = 0
        curr_route = solution_vehicle_x["route"].to_arrow().to_pylist()
        for i in range(len(curr_route) - 1):
            travel_time = df_time_matrix[curr_route[i]][curr_route[i + 1]]
            arrival_time += (
                travel_time + v_service_times[assign][curr_route[i]]
            )
            arrival_time = max(
                arrival_time, df_earliest_time[curr_route[i + 1]]
            )
            computed_cost += df_distance_matrix[curr_route[i]][
                curr_route[i + 1]
            ]
            assert arrival_time <= df_latest_time[curr_route[i + 1]]
        assert abs(vehicle_x_total_time - arrival_time) < th
    assert abs(routing_solution.get_total_objective() - computed_cost) < th


def test_vehicle_dependent_service_times():
    """
    Test mixed fleet service times
    """

    costs = cudf.DataFrame(
        {
            0: [0, 3, 4, 5, 2],
            1: [1, 0, 3, 2, 7],
            2: [10, 5, 0, 2, 9],
            3: [3, 11, 1, 0, 6],
            4: [5, 3, 8, 6, 0],
        },
        dtype=np.float32,
    )
    vehicle_num = 2
    earliest_time = cudf.Series([0, 0, 0, 0, 0], dtype=np.int32)
    latest_time = cudf.Series(
        [60000, 60000, 60000, 60000, 60000], dtype=np.int32
    )
    service_times = {
        0: [0, 5, 55, 3, 1],
        1: [0, 2, 100, 46, 96],
    }

    pickup_orders = cudf.Series([1, 2])
    delivery_orders = cudf.Series([3, 4])

    d = routing.DataModel(costs.shape[0], vehicle_num)
    d.add_cost_matrix(costs)
    d.set_pickup_delivery_pairs(pickup_orders, delivery_orders)
    d.set_order_time_windows(earliest_time, latest_time)
    for vehicle_id, v_service_times in service_times.items():
        d.set_order_service_times(cudf.Series(v_service_times), vehicle_id)
    d.set_min_vehicles(2)

    settings = routing.SolverSettings()
    settings.set_time_limit(2)

    routing_solution = routing.Solve(d, settings)
    cu_status = routing_solution.get_status()
    assert cu_status == 0
    check_cuopt_solution(
        routing_solution,
        costs,
        costs,
        earliest_time,
        latest_time,
        service_times,
    )
