# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf

from cuopt import routing


def test_prize_collection():
    """
    Test prize collection
    """

    costs = cudf.DataFrame(
        {
            0: [0, 1, 1, 1],
            1: [1, 0, 1, 1],
            2: [
                1,
                1,
                0,
                1,
            ],
            3: [1, 1, 1, 0],
        }
    ).astype(np.float32)

    vehicle_num = 1
    order_loc = cudf.Series([1, 2, 3])
    order_prizes = cudf.Series([5.0, 10.0, 5.0]).astype(np.int32)
    cap = cudf.Series([2])
    dem = cudf.Series([1, 2, 1])
    earliest = cudf.Series([0, 0, 0])
    latest = cudf.Series([1000, 1000, 1000])

    d = routing.DataModel(costs.shape[0], vehicle_num, len(order_loc))
    d.add_cost_matrix(costs)
    d.set_order_locations(order_loc)
    d.set_order_prizes(order_prizes)
    d.add_capacity_dimension("dem", dem, cap)
    d.set_order_time_windows(earliest, latest)

    s = routing.SolverSettings()
    s.set_time_limit(2)

    routing_solution = routing.Solve(d, s)
    cu_status = routing_solution.get_status()
    objectives = routing_solution.get_objective_values()

    assert cu_status == 0
    assert routing_solution.get_total_objective() == -8.0
    assert objectives[routing.Objective.PRIZE] == -10.0
    assert objectives[routing.Objective.COST] == 2.0


def test_min_vehicles():
    """
    Test min vehicles when prize collection is enabled
    """
    cost_1 = cudf.DataFrame(
        [
            [0, 5, 4, 3, 5],
            [5, 0, 6, 4, 3],
            [4, 8, 0, 4, 2],
            [1, 4, 3, 0, 4],
            [3, 3, 5, 6, 0],
        ]
    ).astype(np.float32)

    time_1 = cudf.DataFrame(
        [
            [0, 10, 8, 6, 10],
            [10, 0, 12, 8, 6],
            [8, 16, 0, 8, 4],
            [2, 8, 6, 0, 8],
            [6, 6, 10, 12, 0],
        ]
    ).astype(np.float32)

    cost_2 = cudf.DataFrame(
        [
            [0, 3, 2, 2, 4],
            [4, 0, 5, 3, 2],
            [3, 7, 0, 1, 1],
            [1, 2, 2, 0, 3],
            [2, 2, 3, 4, 0],
        ]
    ).astype(np.float32)

    time_2 = cudf.DataFrame(
        [
            [0, 6, 4, 4, 8],
            [8, 0, 10, 6, 4],
            [6, 14, 0, 2, 2],
            [2, 4, 4, 0, 6],
            [4, 4, 6, 8, 0],
        ]
    ).astype(np.float32)

    vehicle_start_loc = cudf.Series([0, 1, 0, 1, 0])
    vehicle_end_loc = cudf.Series([0, 1, 1, 0, 0])

    vehicle_types = cudf.Series([1, 1, 2, 2, 2])
    vehicle_cap = cudf.Series([30, 30, 10, 10, 10])

    vehicle_start = cudf.Series([0, 5, 0, 20, 20])
    vehicle_end = cudf.Series([80, 80, 100, 100, 100])

    vehicle_break_start = cudf.Series([20, 20, 20, 20, 20])
    vehicle_break_end = cudf.Series([25, 25, 25, 25, 25])
    vehicle_break_duration = cudf.Series([1, 1, 1, 1, 1])

    vehicle_max_costs = cudf.Series([100, 100, 100, 100, 100]).astype(
        np.float32
    )
    vehicle_max_times = cudf.Series([120, 120, 120, 120, 120]).astype(
        np.float32
    )

    order_loc = cudf.Series([1, 2, 3, 4])
    demand = cudf.Series([3, 4, 30, 3])

    task_start = cudf.Series([3, 5, 1, 4])
    task_end = cudf.Series([20, 30, 20, 40])
    serv = cudf.Series([3, 1, 8, 4])
    prizes = cudf.Series([4, 4, 15, 3])

    dm = routing.DataModel(cost_1.shape[0], len(vehicle_types), len(order_loc))

    # Cost and Time
    dm.add_cost_matrix(cost_1, 1)
    dm.add_cost_matrix(cost_2, 2)
    dm.add_transit_time_matrix(time_1, 1)
    dm.add_transit_time_matrix(time_2, 2)
    dm.set_vehicle_types(vehicle_types)
    dm.set_vehicle_locations(vehicle_start_loc, vehicle_end_loc)
    dm.set_vehicle_time_windows(vehicle_start, vehicle_end)
    dm.add_break_dimension(
        vehicle_break_start, vehicle_break_end, vehicle_break_duration
    )
    dm.set_vehicle_max_costs(vehicle_max_costs)
    dm.set_vehicle_max_times(vehicle_max_times)
    dm.add_vehicle_order_match(3, cudf.Series([0, 3]))
    dm.set_min_vehicles(2)
    dm.set_order_locations(order_loc)
    dm.add_capacity_dimension("1", demand, vehicle_cap)
    dm.set_order_time_windows(task_start, task_end)
    dm.set_order_service_times(serv)
    dm.add_order_vehicle_match(3, cudf.Series([3]))
    dm.add_order_vehicle_match(0, cudf.Series([3]))
    dm.set_order_prizes(prizes)

    sol_set = routing.SolverSettings()

    sol_set.set_time_limit(15)

    sol = routing.Solve(dm, sol_set)

    assert sol.get_status() == 0
    assert sol.get_vehicle_count() >= 2


def test_zero_prize():
    """
    Test prize collection when prize objective is zero
    """

    costs = cudf.DataFrame(
        {
            0: [0, 1, 1, 1],
            1: [1, 0, 1, 1],
            2: [
                1,
                1,
                0,
                1,
            ],
            3: [1, 1, 1, 0],
        }
    ).astype(np.float32)

    vehicle_num = 1
    order_loc = cudf.Series([1, 2, 3])
    order_prizes = cudf.Series([5.0, 10.0, 5.0]).astype(np.int32)

    # Set capacity such that there is no feasible solution
    cap = cudf.Series([2])
    dem = cudf.Series([1, 1, 1])

    d = routing.DataModel(costs.shape[0], vehicle_num, len(order_loc))
    d.add_cost_matrix(costs)
    d.set_order_locations(order_loc)
    d.set_order_prizes(order_prizes)
    d.add_capacity_dimension("dem", dem, cap)
    d.set_objective_function(
        cudf.Series([routing.Objective.PRIZE]), cudf.Series([0])
    )

    s = routing.SolverSettings()
    s.set_time_limit(2)

    routing_solution = routing.Solve(d, s)
    cu_status = routing_solution.get_status()

    # Solution should be infeasible
    assert cu_status == 1


def test_no_feasible_task():
    """
    This is a corner case test when none of the task is feasible
    """

    costs = cudf.DataFrame(
        {
            0: [0, 10, 10, 10],
            1: [10, 0, 10, 10],
            2: [
                10,
                10,
                0,
                10,
            ],
            3: [10, 10, 10, 0],
        }
    ).astype(np.float32)

    vehicle_num = 4
    vehicle_start_times = cudf.Series([0, 0, 0, 0]).astype(np.int32)
    vehicle_return_times = cudf.Series([25, 22, 26, 29]).astype(np.int32)

    order_loc = cudf.Series([1, 2])
    order_prizes = cudf.Series([5.0, 5.0]).astype(np.int32)
    order_service_times = cudf.Series([10, 10]).astype(np.int32)

    d = routing.DataModel(costs.shape[0], vehicle_num, len(order_loc))
    d.add_cost_matrix(costs)
    d.add_transit_time_matrix(costs)

    d.set_vehicle_time_windows(vehicle_start_times, vehicle_return_times)

    d.set_order_locations(order_loc)
    d.set_order_service_times(order_service_times)
    d.set_order_prizes(order_prizes)

    s = routing.SolverSettings()
    s.set_time_limit(2)

    routing_solution = routing.Solve(d, s)
    cu_status = routing_solution.get_status()

    assert cu_status == 0
    assert routing_solution.get_total_objective() == 0.0
    assert routing_solution.get_vehicle_count() == 0
