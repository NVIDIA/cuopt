# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np

import cudf

from cuopt import routing
from cuopt.routing import utils


def test_solomon():
    utils.convert_solomon_inp_file_to_yaml(
        utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.txt"
    )
    service_list, vehicle_capacity, vehicle_num = utils.create_from_yaml_file(
        utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.yaml"
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
