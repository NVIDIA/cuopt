# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import cudf

from cuopt import routing


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


def test_lex_smoke():
    cost_matrix = cudf.DataFrame(
        [
            [0.0, 6.0, 4.0, 6.0],
            [6.0, 0.0, 4.0, 6.0],
            [4.0, 4.0, 0.0, 4.0],
            [6.0, 6.0, 4.0, 0.0],
        ]
    )
    vehicle_start = cudf.Series([0, 0])
    vehicle_cap = cudf.Series([2, 2])
    vehicle_eal = cudf.Series([0, 0])
    vehicle_lat = cudf.Series([100, 100])
    task_locations = cudf.Series([1, 2, 3, 3, 2, 1, 2, 3, 0, 2, 1, 0])
    demand = cudf.Series([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1])
    task_time_eal = cudf.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    task_time_latest = cudf.Series(
        [10, 20, 30, 10, 20, 30, 45, 45, 45, 45, 45, 45]
    )
    task_serv = cudf.Series([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    pick_ind = cudf.Series([0, 1, 2, 3, 4, 5])
    del_ind = cudf.Series([6, 7, 8, 9, 10, 11])
    dm = routing.DataModel(
        cost_matrix.shape[0], len(vehicle_start), len(task_locations)
    )
    dm.add_cost_matrix(cost_matrix)
    dm.set_order_locations(task_locations)
    dm.add_capacity_dimension("1", demand, vehicle_cap)
    dm.set_vehicle_time_windows(vehicle_eal, vehicle_lat)
    dm.set_order_time_windows(task_time_eal, task_time_latest)
    dm.set_order_service_times(task_serv)
    dm.set_pickup_delivery_pairs(pick_ind, del_ind)
    sol_set = routing.SolverSettings()
    sol = routing.Solve(dm, sol_set)
    assert sol.get_status() == 0
