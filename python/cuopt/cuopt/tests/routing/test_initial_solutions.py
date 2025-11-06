# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import pytest

import cudf

from cuopt import routing


class TestOption(Enum):
    __test__ = False
    VRP = 0
    SKIP_DEPOTS = 1
    PDP = 2
    BREAKS = 3
    PRIZE = 4


def get_initial_solutions(routing_solution, n_initial_sols=5):
    initial_sol = routing_solution.get_route()
    sol_offsets = [0]
    vehicle_ids = cudf.Series()
    routes = cudf.Series()
    types = cudf.Series()
    # simply expand the same solution for convenience
    for i in range(0, n_initial_sols):
        vehicle_ids = cudf.concat([vehicle_ids, initial_sol["truck_id"]])
        routes = cudf.concat([routes, initial_sol["route"]])
        types = cudf.concat([types, initial_sol["type"]])
        sol_offsets.append(sol_offsets[i] + initial_sol["route"].shape[0])
    sol_offsets = cudf.Series(sol_offsets)
    return vehicle_ids, routes, types, sol_offsets


@pytest.mark.parametrize(
    "flag",
    [
        TestOption.PRIZE,
        TestOption.VRP,
        TestOption.PDP,
        TestOption.SKIP_DEPOTS,
        TestOption.BREAKS,
    ],
)
def test_initial_solutions(flag):
    """
    Test mixed fleet max cost per vehicle
    """

    costs = cudf.DataFrame(
        {
            0: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            1: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            2: [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            3: [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            4: [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            5: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            6: [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            7: [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            8: [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            9: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            10: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )

    vehicle_num = 5
    n_orders = 10
    demand = cudf.Series([1] * n_orders)
    order_loc = cudf.Series(list(range(1, n_orders + 1)))
    capacities = cudf.Series([2] * vehicle_num)
    d = routing.DataModel(costs.shape[0], vehicle_num, n_orders)
    d.add_cost_matrix(costs)
    if flag != TestOption.PDP:
        d.add_capacity_dimension("demand", demand, capacities)
    d.set_order_locations(order_loc)
    if flag == TestOption.SKIP_DEPOTS:
        d.set_skip_first_trips(cudf.Series([1, 1, 1, 1, 1]))
        d.set_drop_return_trips(cudf.Series([1, 1, 1, 1, 1]))
    if flag == TestOption.BREAKS:
        d.add_break_dimension(
            cudf.Series([0] * vehicle_num),
            cudf.Series([1000] * vehicle_num),
            cudf.Series([0] * vehicle_num),
        )
    if flag == TestOption.PDP:
        d.set_pickup_delivery_pairs(
            cudf.Series([0, 1, 2, 3, 4]), cudf.Series([5, 6, 7, 8, 9])
        )
        d.add_capacity_dimension(
            "demand",
            cudf.Series([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]),
            capacities,
        )
    if flag == TestOption.PRIZE:
        d.set_order_prizes(cudf.Series([1] * n_orders))
        # We do not have to set this but it shows we can add
        # missing orders if given an incomplete solution
        d.set_objective_function(
            cudf.Series([routing.Objective.PRIZE, routing.Objective.COST]),
            cudf.Series([2**32, 1]),
        )

    s = routing.SolverSettings()
    s.set_time_limit(2)
    routing_solution = routing.Solve(d, s)

    cu_status = routing_solution.get_status()
    assert cu_status == 0
    original_cost = routing_solution.get_total_objective()

    vehicle_ids, routes, types, sol_offsets = get_initial_solutions(
        routing_solution
    )
    if flag == TestOption.PRIZE:
        vehicle_ids = cudf.Series([0, 0, 0, 0])
        routes = cudf.Series([0, 1, 2, 0])
        types = cudf.Series(["Depot", "Delivery", "Delivery", "Depot"])
        sol_offsets = cudf.Series([0, 4])

    d.add_initial_solutions(vehicle_ids, routes, types, sol_offsets)
    s.set_time_limit(1)
    routing_solution = routing.Solve(d, s)

    cu_status = routing_solution.get_status()
    assert cu_status == 0

    new_cost = routing_solution.get_total_objective()
    assert new_cost <= original_cost
