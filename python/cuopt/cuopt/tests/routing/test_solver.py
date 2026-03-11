# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np

import cudf

from cuopt import routing
from cuopt.routing import utils

import math

SOLOMON_DATASETS_PATH = os.path.join(
    utils.RAPIDS_DATASET_ROOT_DIR, "solomon/In/"
)


def test_solomon():
    SOLOMON_DATASET = "r107.txt"
    SOLOMON_YAML = "r107.yaml"
    utils.convert_solomon_inp_file_to_yaml(
        SOLOMON_DATASETS_PATH + SOLOMON_DATASET
    )
    service_list, vehicle_capacity, vehicle_num = utils.create_from_yaml_file(
        SOLOMON_DATASETS_PATH + SOLOMON_YAML
    )

    distances = utils.build_matrix(service_list)
    distances = distances.astype(np.float32)

    nodes = service_list["demand"].shape[0]
    d = routing.DataModel(nodes, vehicle_num)
    d.add_cost_matrix(distances)

    demand = service_list["demand"].astype(np.int32)
    capacity_list = vehicle_capacity
    capacity_series = cudf.Series(capacity_list)
    d.add_capacity_dimension("demand", demand, capacity_series)

    earliest = service_list["earliest_time"].astype(np.int32)
    latest = service_list["latest_time"].astype(np.int32)
    service = service_list["service_time"].astype(np.int32)
    d.set_order_time_windows(earliest, latest)
    d.set_order_service_times(service)

    s = routing.SolverSettings()
    # set it back to nodes/3 once issue with ARM is resolved
    s.set_time_limit(nodes)

    routing_solution = routing.Solve(d, s)

    vehicle_size = routing_solution.get_vehicle_count()
    final_cost = routing_solution.get_total_objective()
    cu_status = routing_solution.get_status()

    ref_cost = 1087.15
    assert cu_status == 0
    assert vehicle_size <= 12
    if vehicle_size == 11:
        assert math.fabs((final_cost - ref_cost) / ref_cost) < 0.1


def test_pdptw():
    """
    Solve a small PDPTW: 5 locations (depot 0, pickups 1–2, deliveries 3–4),
    2 vehicles, 2 requests. Pickup must be visited before its delivery.
    """
    # Locations: 0 = depot, 1 = pickup A, 2 = pickup B, 3 = delivery A, 4 = delivery B
    n_locations = 5
    n_vehicles = 2

    # Cost/distance matrix (symmetric, integer)
    costs = cudf.DataFrame(
        {
            0: [0, 2, 3, 4, 5],
            1: [2, 0, 2, 3, 4],
            2: [3, 2, 0, 4, 3],
            3: [4, 3, 4, 0, 2],
            4: [5, 4, 3, 2, 0],
        },
        dtype=np.float32,
    )
    # Use same matrix for transit time
    times = costs.astype(np.float32)

    # Pickup-delivery pairs: (order index) pickup 1 -> delivery 3, pickup 2 -> delivery 4
    pickup_indices = cudf.Series([1, 2], dtype=np.int32)
    delivery_indices = cudf.Series([3, 4], dtype=np.int32)

    # Demand: depot 0, pickups +1, deliveries -1
    demand = cudf.Series([0, 1, 1, -1, -1], dtype=np.int32)
    capacities = cudf.Series([2, 2], dtype=np.int32)  # capacity 2 per vehicle

    # Time windows [earliest, latest] per location (wide enough to be feasible)
    earliest = cudf.Series([0, 0, 0, 0, 0], dtype=np.int32)
    latest = cudf.Series([100, 100, 100, 100, 100], dtype=np.int32)
    service_times = cudf.Series([0, 1, 1, 1, 1], dtype=np.int32)

    dm = routing.DataModel(n_locations, n_vehicles)
    dm.add_cost_matrix(costs)
    dm.add_transit_time_matrix(times)
    dm.set_pickup_delivery_pairs(pickup_indices, delivery_indices)
    dm.add_capacity_dimension("demand", demand, capacities)
    dm.set_order_time_windows(earliest, latest)
    dm.set_order_service_times(service_times)

    # Getter checks: pickup/delivery pairs and transit time matrix
    ret_pickup, ret_delivery = dm.get_pickup_delivery_pairs()
    assert (ret_pickup == pickup_indices).all(), (
        "get_pickup_delivery_pairs pickup mismatch"
    )
    assert (ret_delivery == delivery_indices).all(), (
        "get_pickup_delivery_pairs delivery mismatch"
    )
    ret_transit = dm.get_transit_time_matrix(0)
    assert cudf.DataFrame(ret_transit).equals(times), (
        "get_transit_time_matrix mismatch"
    )

    settings = routing.SolverSettings()
    settings.set_time_limit(10)

    solution = routing.Solve(dm, settings)
    status = solution.get_status()

    assert status == 0, (
        f"Expected status 0, got {status}: {solution.get_message()}"
    )
    assert solution.get_vehicle_count() >= 1
    # Exercise Assignment getters (return type / no raise)
    assert isinstance(solution.get_accepted_solutions(), cudf.Series)
    assert isinstance(solution.get_infeasible_orders(), cudf.Series)
    assert solution.get_vehicle_count() <= n_vehicles

    # Check that each route respects pickup-before-delivery (order indices 1 before 3, 2 before 4)
    route_df = solution.get_route()
    for truck_id in route_df["truck_id"].unique().to_arrow().to_pylist():
        vehicle_route = route_df[route_df["truck_id"] == truck_id]
        route_locs = vehicle_route["route"].to_arrow().to_pylist()
        idx_1 = route_locs.index(1) if 1 in route_locs else -1
        idx_2 = route_locs.index(2) if 2 in route_locs else -1
        idx_3 = route_locs.index(3) if 3 in route_locs else -1
        idx_4 = route_locs.index(4) if 4 in route_locs else -1
        if idx_1 >= 0 and idx_3 >= 0:
            assert idx_1 < idx_3, "Pickup 1 must be before delivery 3"
        if idx_2 >= 0 and idx_4 >= 0:
            assert idx_2 < idx_4, "Pickup 2 must be before delivery 4"

    # Optional: basic objective check
    total_cost = solution.get_total_objective()
    assert total_cost == 13.0


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
