# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    transit_times = distances.copy(deep=True)

    nodes = service_list["demand"].shape[0]
    d = routing.DataModel(nodes, vehicle_num)
    d.add_cost_matrix(distances)
    d.add_transit_time_matrix(transit_times)

    demand = service_list["demand"].astype(np.int32)
    capacity_list = [vehicle_capacity] * vehicle_num
    capacity_series = cudf.Series(capacity_list)
    d.add_capacity_dimension("demand", demand, capacity_series)

    earliest = service_list["earliest_time"].astype(np.int32)
    latest = service_list["latest_time"].astype(np.int32)
    service = service_list["service_time"].astype(np.int32)
    d.set_order_time_windows(earliest, latest)
    d.set_order_service_times(service)

    ret_distances = d.get_cost_matrix()
    ret_transit_times = d.get_transit_time_matrix(0)
    ret_vehicle_num = d.get_fleet_size()
    ret_num_orders = d.get_num_orders()
    ret_capacity_dimensions = d.get_capacity_dimensions()
    ret_time_windows = d.get_order_time_windows()
    ret_service_time = d.get_order_service_times()

    assert cudf.DataFrame(ret_distances).equals(distances)
    assert cudf.DataFrame(ret_transit_times).equals(transit_times)
    assert ret_vehicle_num == vehicle_num
    assert ret_num_orders == nodes
    assert (ret_capacity_dimensions["demand"]["demand"] == demand).all()
    assert (
        ret_capacity_dimensions["demand"]["capacity"] == capacity_series
    ).all()
    assert (ret_time_windows[0] == earliest).all()
    assert (ret_time_windows[1] == latest).all()
    assert (ret_service_time == service).all()


def test_objective_function():
    d = utils.create_data_model(filename, run_nodes=10)

    obj = routing.Objective
    objectives = cudf.Series([obj.COST, obj.VARIANCE_ROUTE_SIZE])
    objective_weights = cudf.Series([1, 10]).astype(np.float32)
    d.set_objective_function(objectives, objective_weights)
    ret_objectives, ret_objective_weights = d.get_objective_function()

    assert (objectives == ret_objectives).all()
    assert (objective_weights == ret_objective_weights).all()


def test_multi_cost_and_transit_matrices_getters():
    """Assert getters return correct multi vehicle-type cost and transit matrices."""
    cost_1 = cudf.DataFrame(
        [[0, 4, 4], [4, 0, 4], [4, 4, 0]], dtype=np.float32
    )
    time_1 = cudf.DataFrame(
        [[0, 50, 50], [50, 0, 50], [50, 50, 0]], dtype=np.float32
    )
    cost_2 = cudf.DataFrame(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32
    )
    time_2 = cudf.DataFrame(
        [[0, 10, 10], [10, 0, 10], [10, 10, 0]], dtype=np.float32
    )
    dm = routing.DataModel(3, 2)
    dm.add_cost_matrix(cost_1, 1)
    dm.add_transit_time_matrix(time_1, 1)
    dm.add_cost_matrix(cost_2, 2)
    dm.add_transit_time_matrix(time_2, 2)
    dm.set_vehicle_types(cudf.Series([1, 2]))

    assert cudf.DataFrame(dm.get_cost_matrix(1)).equals(cost_1)
    assert cudf.DataFrame(dm.get_cost_matrix(2)).equals(cost_2)
    assert cudf.DataFrame(dm.get_transit_time_matrix(1)).equals(time_1)
    assert cudf.DataFrame(dm.get_transit_time_matrix(2)).equals(time_2)
