# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf

from cuopt import routing


def test_vehicle_max_costs():
    """
    Test mixed fleet max cost per vehicle
    """

    costs = cudf.DataFrame(
        {
            0: [0, 3, 4, 5, 2],
            1: [1, 0, 3, 2, 7],
            2: [10, 5, 0, 2, 9],
            3: [3, 11, 1, 0, 6],
            4: [5, 3, 8, 6, 0],
        }
    )

    vehicle_num = 4
    vehicle_max_costs = cudf.Series([11, 12, 11, 15])

    d = routing.DataModel(costs.shape[0], vehicle_num)
    d.add_cost_matrix(costs)
    d.set_vehicle_max_costs(vehicle_max_costs)

    s = routing.SolverSettings()
    s.set_time_limit(1)

    routing_solution = routing.Solve(d, s)
    cu_status = routing_solution.get_status()
    solution_cudf = routing_solution.get_route()

    assert cu_status == 0

    for i, assign in enumerate(
        solution_cudf["truck_id"].unique().to_arrow().to_pylist()
    ):
        curr_route_dist = 0
        solution_vehicle_x = solution_cudf[solution_cudf["truck_id"] == assign]
        h_route = solution_vehicle_x["route"].to_arrow().to_pylist()
        route_len = len(h_route)
        for j in range(route_len - 1):
            curr_route_dist += costs.iloc[h_route[j], h_route[j + 1]]

        assert curr_route_dist < vehicle_max_costs[assign] + 0.001
