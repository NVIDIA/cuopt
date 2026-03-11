# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf

from cuopt import routing
from cuopt.routing import utils
from cuopt.routing.vehicle_routing_wrapper import ErrorStatus

filename = utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.txt"


# ----- Vehicle types -----


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
    assert dm.get_min_vehicles() == 2
    assert (dm.get_vehicle_types() == vehicle_types).all()

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


# ----- Vehicle fixed costs -----


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
    assert (d.get_vehicle_fixed_costs() == vehicle_fixed_costs).all()

    s = routing.SolverSettings()
    s.set_time_limit(3)

    routing_solution = routing.Solve(d, s)
    cu_status = routing_solution.get_status()
    objectives = routing_solution.get_objective_values()

    assert cu_status == 0
    assert routing_solution.get_total_objective() == 49
    assert objectives[routing.Objective.VEHICLE_FIXED_COST] == 35
    assert objectives[routing.Objective.COST] == 14


# ----- Vehicle max cost -----


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
    assert (d.get_vehicle_max_costs() == vehicle_max_costs).all()

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


# ----- Vehicle max time -----


def test_vehicle_max_times_fail():
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
    vehicle_num = 4
    vehicle_max_times = cudf.Series([100, 30, 50, 70], dtype=np.float32)

    d = routing.DataModel(costs.shape[0], vehicle_num)
    d.add_cost_matrix(costs)
    d.set_vehicle_max_times(vehicle_max_times)

    s = routing.SolverSettings()
    s.set_time_limit(1)

    routing_solution = routing.Solve(d, s)
    assert routing_solution.get_error_status() == ErrorStatus.ValidationError
    err_msg = routing_solution.get_error_message().decode()
    assert (
        "Time matrix should be set in order to use vehicle max time constraints"
        in err_msg
    )


def test_vehicle_max_times():
    """
    Test mixed fleet max time per vehicle
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
    times = costs * 10

    vehicle_num = 4
    vehicle_max_times = cudf.Series([100, 30, 50, 70], dtype=np.float32)

    d = routing.DataModel(costs.shape[0], vehicle_num)
    d.add_cost_matrix(costs)
    d.add_transit_time_matrix(times)
    d.set_vehicle_max_times(vehicle_max_times)
    assert (d.get_vehicle_max_times() == vehicle_max_times).all()

    s = routing.SolverSettings()
    s.set_time_limit(10)

    routing_solution = routing.Solve(d, s)
    cu_status = routing_solution.get_status()
    solution_cudf = routing_solution.get_route()

    assert cu_status == 0

    for i, assign in enumerate(
        solution_cudf["truck_id"].unique().to_arrow().to_pylist()
    ):
        curr_route_time = 0
        solution_vehicle_x = solution_cudf[solution_cudf["truck_id"] == assign]
        h_route = solution_vehicle_x["route"].to_arrow().to_pylist()
        route_len = len(h_route)
        for j in range(route_len - 1):
            curr_route_time += times.iloc[h_route[j], h_route[j + 1]]

        assert curr_route_time < vehicle_max_times[assign] + 0.001


# ----- Order / vehicle match -----


def test_order_to_vehicle_match():
    n_vehicles = 3
    n_locations = 4
    time_mat = [[0, 1, 5, 2], [2, 0, 7, 4], [1, 5, 0, 9], [5, 6, 2, 0]]

    order_vehicle_match = {1: [0], 3: [0], 2: [1]}

    d = routing.DataModel(n_locations, n_vehicles)
    d.add_cost_matrix(cudf.DataFrame(time_mat))

    for order, vehicles in order_vehicle_match.items():
        d.add_order_vehicle_match(order, cudf.Series(vehicles))

    ret_order_vehicle = d.get_order_vehicle_match()
    assert set(ret_order_vehicle.keys()) == set(order_vehicle_match.keys())
    for order, vehicles in order_vehicle_match.items():
        assert ret_order_vehicle[order].to_arrow().to_pylist() == vehicles

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

    ret_vehicle_order = d.get_vehicle_order_match()
    assert set(ret_vehicle_order.keys()) == set(vehicle_order_match.keys())
    for vehicle, orders in vehicle_order_match.items():
        assert ret_vehicle_order[vehicle].to_arrow().to_pylist() == orders

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

    order_loc = cudf.Series([1, 2, 3])
    d.set_order_locations(order_loc)
    for order, vehicles in order_vehicle_match.items():
        d.add_order_vehicle_match(order, cudf.Series(vehicles))
    assert (d.get_order_locations() == order_loc).all()

    s = routing.SolverSettings()
    s.set_time_limit(5)

    routing_solution = routing.Solve(d, s)
    vehicle_count = routing_solution.get_vehicle_count()
    cu_status = routing_solution.get_status()

    assert cu_status == 0
    assert vehicle_count == 1


# ----- Vehicle time windows and locations -----


def test_time_windows():
    vehicle_num = 5
    d = utils.create_data_model(
        filename, num_vehicles=vehicle_num * 2, run_nodes=10
    )

    vehicle_earliest = []
    vehicle_latest = []
    latest_time = d.get_order_time_windows()[1].max()
    buffer_time = 50.0  # Time to travel back to or from the depot
    for i in range(vehicle_num):
        vehicle_earliest.append(0)
        vehicle_latest.append(latest_time / 2 + buffer_time)
    for i in range(vehicle_num):
        vehicle_earliest.append(latest_time / 2 - buffer_time)
        vehicle_latest.append(latest_time + buffer_time)
    d.set_vehicle_time_windows(
        cudf.Series(vehicle_earliest).astype(np.int32),
        cudf.Series(vehicle_latest).astype(np.int32),
    )

    s = routing.SolverSettings()
    s.set_time_limit(10)
    routing_solution = routing.Solve(d, s)

    ret_vehicle_time_windows = d.get_vehicle_time_windows()
    assert (ret_vehicle_time_windows[0] == cudf.Series(vehicle_earliest)).all()
    assert (ret_vehicle_time_windows[1] == cudf.Series(vehicle_latest)).all()

    assert routing_solution.get_status() == 0

    routes = routing_solution.get_route()
    truck_ids = routing_solution.get_route()["truck_id"].unique()

    for i in range(len(truck_ids)):
        truck_id = truck_ids.iloc[i]
        vehicle_route = routes[routes["truck_id"] == truck_id]
        assert (
            vehicle_route["arrival_stamp"].iloc[0]
            >= vehicle_earliest[truck_id]
        )
        assert (
            vehicle_route["arrival_stamp"].iloc[-1] <= vehicle_latest[truck_id]
        )


def test_vehicle_locations():
    d = utils.create_data_model(filename, run_nodes=10)
    num_vehicles = d.get_fleet_size()
    v_start_locations = cudf.Series([4] * num_vehicles)
    v_end_locations = cudf.Series([10] * num_vehicles)
    d.set_vehicle_locations(v_start_locations, v_end_locations)
    ret_start_locations, ret_end_locations = d.get_vehicle_locations()

    assert (v_start_locations == ret_start_locations).all()
    assert (v_end_locations == ret_end_locations).all()

    s = routing.SolverSettings()
    s.set_time_limit(10)
    routing_solution = routing.Solve(d, s)

    routes = routing_solution.get_route()
    truck_ids = routing_solution.get_route()["truck_id"].unique()

    for i in range(len(truck_ids)):
        truck_id = truck_ids.iloc[i]
        vehicle_route = routes[routes["truck_id"] == truck_id]
        assert vehicle_route["location"].iloc[0] == 4
        assert vehicle_route["location"].iloc[-1] == 10


# ----- Vehicle breaks -----


def test_heterogenous_breaks():
    vehicle_num = 5
    run_nodes = 20
    d = utils.create_data_model(
        filename, run_nodes=run_nodes, num_vehicles=vehicle_num
    )

    """
    Half of vehicles have three breaks and the remaining half have two breaks.
    Break locations are also different. First set of vehicles have specified
    subset of locations while the second set of vehicles have default, i.e. any
    location can be a break
    """
    num_breaks_1 = 2
    num_v_type_1 = int(vehicle_num / 2)
    break_times_1 = [[90, 100], [150, 170]]
    break_durations_1 = [15, 15]
    break_locations_1 = cudf.Series([4 * i for i in range(1, 5)])

    num_breaks_2 = 3
    num_v_type_2 = vehicle_num - num_v_type_1
    break_times_2 = [[40, 50], [110, 120], [160, 170]]
    break_durations_2 = [10, 10, 10]

    for i in range(num_v_type_1):
        for b in range(num_breaks_1):
            d.add_vehicle_break(
                i,
                break_times_1[b][0],
                break_times_1[b][1],
                break_durations_1[b],
                break_locations_1,
            )

    for i in range(num_v_type_2):
        for b in range(num_breaks_2):
            d.add_vehicle_break(
                i + num_v_type_1,
                break_times_2[b][0],
                break_times_2[b][1],
                break_durations_2[b],
            )

    ret_non_uniform = d.get_non_uniform_breaks()
    assert len(ret_non_uniform) == vehicle_num
    for i in range(num_v_type_1):
        assert len(ret_non_uniform[i]) == num_breaks_1
    for i in range(num_v_type_2):
        assert len(ret_non_uniform[num_v_type_1 + i]) == num_breaks_2

    s = routing.SolverSettings()
    s.set_time_limit(30)
    routing_solution = routing.Solve(d, s)

    # TO DO: Check if breaks are adhered to
    assert routing_solution.get_status() == 0
    counters = {}
    routes = routing_solution.get_route().to_pandas()
    break_locations_1_list = break_locations_1.to_arrow().to_pylist()
    # make sure the break locations are the right ones and
    # the arrival stamps satisfy the break time constraints
    for i in range(routes.shape[0]):
        truck_id = routes["truck_id"][i]
        if truck_id not in counters:
            counters[truck_id] = 0
        if routes["type"][i] == "Break":
            break_dim = routes["route"][i]
            location = routes["location"][i]
            arrival_time = routes["arrival_stamp"][i]
            if truck_id < num_v_type_1:
                assert location in break_locations_1_list
                assert arrival_time >= break_times_1[break_dim][0]
                assert arrival_time <= break_times_1[break_dim][1]
            else:
                assert arrival_time >= break_times_2[break_dim][0]
                assert arrival_time <= break_times_2[break_dim][1]
            counters[truck_id] = counters[truck_id] + 1

    # Make sure the achieved number of breaks is same as the specified
    for truck_id, num_breaks in counters.items():
        if truck_id < num_v_type_1:
            assert num_breaks == num_breaks_1
        else:
            assert num_breaks == num_breaks_2


# ----- Vehicle dependent service times -----


def _check_cuopt_solution(
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
    assert d.get_min_vehicles() == 2

    settings = routing.SolverSettings()
    settings.set_time_limit(2)

    routing_solution = routing.Solve(d, settings)
    cu_status = routing_solution.get_status()
    assert cu_status == 0
    _check_cuopt_solution(
        routing_solution,
        costs,
        costs,
        earliest_time,
        latest_time,
        service_times,
    )


# ----- Vehicle routing breaks and order match -----


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
    assert (dm.get_order_locations() == task_locations).all()

    dm.set_vehicle_types(veh_types)

    break_locations = cudf.Series([1, 2, 3, 4, 5])
    dm.set_break_locations(break_locations)
    dm.add_break_dimension(
        vehicle_break_eal, vehicle_break_lat, vehicle_duration
    )
    assert (dm.get_break_locations() == break_locations).all()
    ret_break_dims = dm.get_break_dimensions()
    assert len(ret_break_dims) == 1
    dim0 = list(ret_break_dims.values())[0]
    assert (dim0["earliest"] == vehicle_break_eal).all()
    assert (dim0["latest"] == vehicle_break_lat).all()
    assert (dim0["duration"] == vehicle_duration).all()

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
