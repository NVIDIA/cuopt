# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import random

import numpy as np

import cudf

from cuopt import routing
from cuopt.routing import utils
from cuopt.routing.re_routing import construct_rerouting_model


def generate_new_order_data(
    distances, new_pairs, start_time, max_demand, max_service_time
):
    """
    Utility function that generates a new batch of orders
    at the already existing locations
    """
    nlocations = distances.shape[0]
    # populate with empty vectors
    new_order_locations = []
    new_order_earliest_time = []
    new_order_latest_time = []
    new_order_service_time = []
    new_order_demand = []
    new_order_pickup_indices = []
    new_order_delivery_indices = []

    distances_h = distances.to_numpy()
    for i in range(new_pairs):
        pickup_loc = random.randint(0, nlocations - 1)
        delivery_loc = random.randint(0, nlocations - 1)

        new_order_pickup_indices.append(2 * i)
        new_order_delivery_indices.append(2 * i + 1)
        new_order_locations.append(pickup_loc)
        new_order_locations.append(delivery_loc)

        d = random.randint(1, max_demand)
        new_order_demand.append(d)
        new_order_demand.append(-d)

        st = random.randint(1, max_service_time)
        new_order_service_time.append(st)
        new_order_service_time.append(st)

        new_order_earliest_time.append(math.ceil(start_time))
        new_order_earliest_time.append(math.ceil(start_time))

        r = random.randint(2, 4)
        dist_from_depot_to_pickup = math.ceil(distances_h[0, pickup_loc])
        latest_time_of_pickup = start_time + dist_from_depot_to_pickup * r

        r = random.randint(2, 4)
        dist_from_pickup_to_del = math.ceil(
            distances_h[pickup_loc, delivery_loc]
        )
        latest_time_of_delivery = (
            latest_time_of_pickup + st + dist_from_pickup_to_del * r
        )

        new_order_latest_time.append(math.ceil(latest_time_of_pickup))
        new_order_latest_time.append(math.ceil(latest_time_of_delivery))

    new_order_data = {}
    new_order_data["order_locations"] = new_order_locations
    new_order_data["earliest_time"] = new_order_earliest_time
    new_order_data["latest_time"] = new_order_latest_time
    new_order_data["service_time"] = new_order_service_time
    new_order_data["pickup_indices"] = new_order_pickup_indices
    new_order_data["delivery_indices"] = new_order_delivery_indices
    new_order_data["demand"] = new_order_demand

    return new_order_data


def test_re_routing():
    utils.convert_solomon_inp_file_to_yaml(
        utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.txt"
    )
    service_list, vehicle_capacity, vehicle_num = utils.create_from_yaml_file(
        utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.yaml"
    )

    # Truncate the model so that we run the test fast
    nodes_to_keep = 21
    nodes_to_remove = [i for i in range(nodes_to_keep, len(service_list))]
    service_list.drop(index=nodes_to_remove, inplace=True)

    # set latest time of first order to 0 so that it is the first one
    # after sorting
    service_list["latest_time"][0] = 0
    service_list = service_list.sort_values(
        by="latest_time", ignore_index=True
    )
    service_list.reset_index(drop=True, inplace=True)

    # set latest time very high so that we can do multiple dynamic optimization
    # instances
    service_list["latest_time"][0] = 2147483647

    if service_list["demand"].shape[0] % 2 == 0:
        n = service_list["demand"].shape[0]
        service_list.drop(index=n - 1, inplace=True, ignore_index=True)

    distances = utils.build_matrix(service_list)
    distances = distances.astype(np.float32)

    nlocations = service_list["demand"].shape[0]
    # drop the depot node
    service_list.drop(index=0, inplace=True)
    service_list.reset_index(drop=True, inplace=True)
    nodes = service_list["demand"].shape[0]

    npairs = int(nodes / 2)
    pickup_indices = [i for i in range(npairs)]
    delivery_indices = [i + npairs for i in range(npairs)]

    order_locations = [i + 1 for i in range(nodes)]
    data_model = routing.DataModel(nlocations, vehicle_num, nodes)
    data_model.add_cost_matrix(distances)

    data_model.set_order_locations(cudf.Series(order_locations))
    data_model.set_pickup_delivery_pairs(
        cudf.Series(pickup_indices), cudf.Series(delivery_indices)
    )

    demand = service_list["demand"].to_arrow().to_pylist()
    for i in range(npairs):
        demand[i + npairs] = -demand[i]
    demand = cudf.Series(demand)

    capacity_list = vehicle_capacity
    capacity_series = cudf.Series(capacity_list)
    data_model.add_capacity_dimension("demand", demand, capacity_series)

    earliest = service_list["earliest_time"].astype(np.int32)
    latest = service_list["latest_time"].astype(np.int32)
    service = service_list["service_time"].astype(np.int32)
    data_model.set_order_time_windows(earliest, latest)
    data_model.set_order_service_times(service)

    data_model.set_drop_return_trips(cudf.Series([1] * vehicle_num))

    solver_settings = routing.SolverSettings()

    routing_solution = routing.Solve(data_model, solver_settings)
    cu_route = routing_solution.get_route()

    # Specify seed for deterministic runs
    random.seed(13)

    for i in range(4):
        # Generate random new order data starting from median of previous batch
        reschedule_time = math.ceil(cu_route["arrival_stamp"].median())
        new_pairs = random.randint(1, 20)
        new_order_data = generate_new_order_data(
            distances, new_pairs, reschedule_time, demand.max(), service.max()
        )

        # Construct re routing model
        reschedule_data_model = construct_rerouting_model(
            data_model, cu_route, reschedule_time, new_order_data, distances
        )

        # Solve the re routed model
        solver_settings = routing.SolverSettings()
        routing_solution = routing.Solve(
            reschedule_data_model, solver_settings
        )

        # update the data model
        data_model = reschedule_data_model

        assert routing_solution.get_status() == 0
        cu_route = routing_solution.get_route()
