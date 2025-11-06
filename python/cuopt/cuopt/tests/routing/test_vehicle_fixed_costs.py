# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf

from cuopt import routing


def test_vehicle_fixed_costs():
    """
    Test mixed fleet fixed cost per vehicle
    """

    costs = cudf.DataFrame(
        {
            0: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            1: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            2: [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            3: [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            4: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            5: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            6: [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            7: [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            8: [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            9: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )

    vehicle_num = 16
    vehicle_fixed_costs = cudf.Series(
        [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 1, 1, 1]
    )
    demand = cudf.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    capacities = cudf.Series([2] * vehicle_num)

    d = routing.DataModel(costs.shape[0], vehicle_num)
    d.add_cost_matrix(costs)
    d.add_capacity_dimension("demand", demand, capacities)
    d.set_vehicle_fixed_costs(vehicle_fixed_costs)

    s = routing.SolverSettings()
    s.set_time_limit(3)

    routing_solution = routing.Solve(d, s)
    routing_solution.display_routes()
    cu_status = routing_solution.get_status()
    objectives = routing_solution.get_objective_values()

    assert cu_status == 0
    assert routing_solution.get_total_objective() == 49
    assert objectives[routing.Objective.VEHICLE_FIXED_COST] == 35
    assert objectives[routing.Objective.COST] == 14
