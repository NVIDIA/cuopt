# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf

from cuopt import routing
from cuopt.routing import utils

filename = utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.txt"
service_list, vehicle_capacity, vehicle_num = utils.create_from_file(filename)


def test_order_constraints():
    distances = utils.build_matrix(service_list)
    distances = distances.astype(np.float32)

    nodes = service_list["demand"].shape[0]
    d = routing.DataModel(nodes, vehicle_num)
    d.add_cost_matrix(distances)

    demand = service_list["demand"].astype(np.int32)
    capacity_list = [vehicle_capacity] * vehicle_num
    capacity_series = cudf.Series(capacity_list)
    d.add_capacity_dimension("demand", demand, capacity_series)

    earliest = service_list["earliest_time"].astype(np.int32)
    latest = service_list["latest_time"].astype(np.int32)
    service = service_list["service_time"].astype(np.int32)
    d.set_order_time_windows(earliest, latest)
    d.set_order_service_times(service)

    s = routing.SolverSettings()

    routing_solution = routing.Solve(d, s)

    ret_distances = d.get_cost_matrix()
    ret_vehicle_num = d.get_fleet_size()
    ret_num_orders = d.get_num_orders()
    ret_capacity_dimensions = d.get_capacity_dimensions()
    ret_time_windows = d.get_order_time_windows()
    ret_service_time = d.get_order_service_times()

    assert cudf.DataFrame(ret_distances).equals(distances)
    assert ret_vehicle_num == vehicle_num
    assert ret_num_orders == nodes
    assert (ret_capacity_dimensions["demand"]["demand"] == demand).all()
    assert (
        ret_capacity_dimensions["demand"]["capacity"] == capacity_series
    ).all()
    assert (ret_time_windows[0] == earliest).all()
    assert (ret_time_windows[1] == latest).all()
    assert (ret_service_time == service).all()

    cu_status = routing_solution.get_status()
    vehicle_size = routing_solution.get_vehicle_count()

    assert cu_status == 0
    assert vehicle_size <= 11


def test_objective_function():
    d = utils.create_data_model(filename, run_nodes=10)

    obj = routing.Objective
    objectives = cudf.Series([obj.COST, obj.VARIANCE_ROUTE_SIZE])
    objective_weights = cudf.Series([1, 10]).astype(np.float32)
    d.set_objective_function(objectives, objective_weights)
    ret_objectives, ret_objective_weights = d.get_objective_function()

    assert (objectives == ret_objectives).all()
    assert (objective_weights == ret_objective_weights).all()
