# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf

from cuopt import routing


def test_order_to_vehicle_match():
    n_vehicles = 3
    n_locations = 4
    time_mat = [[0, 1, 5, 2], [2, 0, 7, 4], [1, 5, 0, 9], [5, 6, 2, 0]]

    order_vehicle_match = {1: [0], 3: [0], 2: [1]}

    d = routing.DataModel(n_locations, n_vehicles)
    d.add_cost_matrix(cudf.DataFrame(time_mat))

    for order, vehicles in order_vehicle_match.items():
        d.add_order_vehicle_match(order, cudf.Series(vehicles))

    s = routing.SolverSettings()
    s.set_time_limit(10)

    routing_solution = routing.Solve(d, s)
    vehicle_count = routing_solution.get_vehicle_count()
    cu_route = routing_solution.get_route()
    cu_status = routing_solution.get_status()

    assert cu_status == 0
    assert vehicle_count == 2

    route_ids = cu_route["route"].to_arrow().to_pylist()
    truck_ids = cu_route["truck_id"].to_arrow().to_pylist()

    for i in range(len(route_ids)):
        order = route_ids[i]
        if order == 1 or order == 3:
            assert truck_ids[i] == 0
        if order == 2:
            assert truck_ids[i] == 1


def test_vehicle_to_order_match():
    """
    A user might have the vehicle to order match instead of
    order to vehicle match, in those cases, we can use
    cudf.DataFrame.transpose to feed the data_model
    """
    n_vehicles = 3
    n_locations = 4
    time_mat = [[0, 1, 5, 2], [2, 0, 7, 4], [1, 5, 0, 9], [5, 6, 2, 0]]

    # Force one vehicle to pick only one order
    vehicle_order_match = {0: [1], 1: [2], 2: [3]}

    d = routing.DataModel(n_locations, n_vehicles)
    d.add_cost_matrix(cudf.DataFrame(time_mat))

    for vehicle, orders in vehicle_order_match.items():
        d.add_vehicle_order_match(vehicle, cudf.Series(orders))

    s = routing.SolverSettings()
    s.set_time_limit(10)

    routing_solution = routing.Solve(d, s)
    vehicle_count = routing_solution.get_vehicle_count()
    cu_route = routing_solution.get_route()
    cu_status = routing_solution.get_status()

    assert cu_status == 0
    assert vehicle_count == 3

    route_ids = cu_route["route"].to_arrow().to_pylist()
    truck_ids = cu_route["truck_id"].to_arrow().to_pylist()

    for i in range(len(route_ids)):
        order = route_ids[i]
        if order > 0:
            assert truck_ids[i] == order - 1


def test_single_vehicle_with_match():
    """
    This is a corner case test when there is only one vehicle present
    """
    n_vehicles = 1
    n_locations = 4
    n_orders = 3
    time_mat = [[0, 1, 5, 2], [2, 0, 7, 4], [1, 5, 0, 9], [5, 6, 2, 0]]

    order_vehicle_match = {0: [0], 1: [0], 2: [0]}

    d = routing.DataModel(n_locations, n_vehicles, n_orders)
    d.add_cost_matrix(cudf.DataFrame(time_mat))

    d.set_order_locations(cudf.Series([1, 2, 3]))
    for order, vehicles in order_vehicle_match.items():
        d.add_order_vehicle_match(order, cudf.Series(vehicles))

    s = routing.SolverSettings()
    s.set_time_limit(5)

    routing_solution = routing.Solve(d, s)
    vehicle_count = routing_solution.get_vehicle_count()
    cu_status = routing_solution.get_status()

    assert cu_status == 0
    assert vehicle_count == 1
