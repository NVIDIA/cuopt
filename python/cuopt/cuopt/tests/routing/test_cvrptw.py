# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest

from cuopt import routing
from cuopt.routing import utils


@pytest.fixture(params=utils.DATASETS_SOLOMON)
def data_(request):
    df, vehicle_capacity, n_vehicles = utils.create_from_file(request.param)
    file_name = request.param
    return df, vehicle_capacity, n_vehicles, file_name


use_time_matrix = [False]
solomon_nodes = [100]


@pytest.fixture(params=solomon_nodes)
def nodes_(request):
    return request.param


@pytest.fixture(params=use_time_matrix)
def use_time_matrix_(request):
    return request.param


def test_cvrptw_dist_mat(data_, nodes_):
    df, vehicle_capacity, n_vehicles, file_name = data_
    # read reference, if it doesn't exists skip the test
    try:
        ref_cost, ref_vehicle = utils.read_ref(file_name, "solomon", nodes_)
    except ValueError:
        pytest.skip("Reference could not be found!")

    print(f"Running file {file_name}...")
    df = df.head(nodes_ + 1)  # get only the relative number of nodes
    distances = utils.build_matrix(df)
    distances = distances.astype(np.float32)
    nodes = df["demand"].shape[0]
    d = routing.DataModel(nodes, n_vehicles)
    d.add_cost_matrix(distances)
    utils.fill_demand(df, d, vehicle_capacity, n_vehicles)
    utils.fill_tw(d, df)

    s = routing.SolverSettings()
    utils.set_limits(s, nodes)

    routing_solution = routing.Solve(d, s)
    final_cost = routing_solution.get_total_objective()
    vehicle_count = routing_solution.get_vehicle_count()
    cu_route = routing_solution.get_route()
    cu_status = routing_solution.get_status()

    # assert cu_status == SolutionStatus.SUCCESS
    # status returns integer instead of enum
    assert cu_status == 0
    # FIXME find better error rates
    # assert (final_cost - ref_cost) / ref_cost < 0.2
    assert vehicle_count - ref_vehicle <= 2
    if vehicle_count == ref_vehicle:
        assert final_cost <= math.ceil(ref_cost * 2.0)
    assert cu_route["route"].unique().count() == nodes_ + 1
    assert cu_route["truck_id"].unique().count() == vehicle_count
