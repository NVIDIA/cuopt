# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf

from cuopt import routing
from cuopt.routing import utils

filename = utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.txt"


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
