# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf

from cuopt import routing


def test_prize_collection():
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
    assert (dm.get_order_prizes() == prizes).all()

    sol_set = routing.SolverSettings()

    sol_set.set_time_limit(15)

    sol = routing.Solve(dm, sol_set)

    objectives = sol.get_objective_values()
    assert sol.get_total_objective() == -13.0
    assert objectives[routing.Objective.PRIZE] == -26.0
    assert objectives[routing.Objective.COST] == 13.0
    assert sol.get_status() == 0
    assert sol.get_vehicle_count() >= 2


# Cost matrix from issue #904 (7 locations: depot 0 + orders 1-6)
_ISSUE_904_COST_MATRIX = [
    [0, 17, 12, 11, 10, 18, 10],
    [16, 0, 15, 11, 19, 15, 16],
    [19, 19, 0, 11, 16, 11, 17],
    [17, 19, 17, 0, 11, 18, 19],
    [10, 19, 19, 19, 0, 17, 15],
    [12, 18, 15, 18, 18, 0, 14],
    [12, 12, 11, 19, 10, 17, 0],
]


def _build_min_vehicles_data_model(vehicle_fixed_costs, min_vehicles=3):
    """Builds min vehicles regression data model"""
    n_locations = 7
    n_vehicles = 3
    n_orders = 6
    dm = routing.DataModel(n_locations, n_vehicles, n_orders)
    dm.add_cost_matrix(
        cudf.DataFrame(_ISSUE_904_COST_MATRIX).astype(np.float32)
    )
    # Capacity 10 lets all 6 orders fit on one vehicle; min_vehicles must still
    # force 3 routes.
    dm.add_capacity_dimension(
        "demand",
        cudf.Series([1] * n_orders, dtype=np.int32),
        cudf.Series([10] * n_vehicles, dtype=np.int32),
    )
    dm.set_order_locations(cudf.Series([1, 2, 3, 4, 5, 6], dtype=np.int32))
    dm.set_vehicle_fixed_costs(
        cudf.Series(vehicle_fixed_costs, dtype=np.float32)
    )
    dm.set_min_vehicles(min_vehicles)
    dm.set_objective_function(
        cudf.Series(
            [routing.Objective.COST, routing.Objective.VEHICLE_FIXED_COST]
        ),
        cudf.Series([1.0, 1.0], dtype=np.float32),
    )
    return dm


@pytest.mark.parametrize(
    "vehicle_fixed_costs",
    [
        [10.0, 20.0, 30.0],  # non-zero fixed costs (the bug case in #904)
        [
            0.0,
            0.0,
            0.0,
        ],  # zero fixed costs (H100 12.2/13.1 compat crashed in debug)
    ],
)
def test_min_vehicles_respected(vehicle_fixed_costs):
    """
    Regression for https://github.com/NVIDIA/cuopt/issues/904.
    Verifies that min_vehicles is respected and no crash occurs, regardless of
    vehicle fixed costs.
    """
    dm = _build_min_vehicles_data_model(
        vehicle_fixed_costs=vehicle_fixed_costs
    )
    ss = routing.SolverSettings()
    ss.set_time_limit(3)

    sol = routing.Solve(dm, ss)

    assert sol.get_status() == 0, sol.get_message()
    assert sol.get_vehicle_count() >= 3
