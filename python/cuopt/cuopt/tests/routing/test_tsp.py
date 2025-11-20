# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest

from cuopt import routing
from cuopt.routing import utils

TSP_PATH = os.path.join(utils.RAPIDS_DATASET_ROOT_DIR, "tsp")
DATASETS_TSP = [
    os.path.join(TSP_PATH, "a280.tsp"),
    os.path.join(TSP_PATH, "tsp225.tsp"),
    os.path.join(TSP_PATH, "ch150.tsp"),
]


@pytest.fixture(params=DATASETS_TSP)
def data_(request):
    df = utils.create_from_file_tsp(request.param)
    file_name = request.param
    return df, file_name


# TO-DO: Remove this skip once the TSP Link is fixed and issue #609 is closed
@pytest.mark.skip(reason="Skipping TSP tests")
def test_tsp(data_):
    df, file_name = data_
    # read reference, if it doesn't exists skip the test
    try:
        ref_cost, ref_vehicle = utils.read_ref_tsp(file_name, "l1_tsp")
    except ValueError:
        pytest.skip("Reference could not be found!")

    print(f"Running file {file_name}...")
    distances = utils.build_matrix(df)
    distances = distances.astype(np.float32)
    nodes = df["vertex"].shape[0]
    d = routing.DataModel(nodes, 1)
    d.add_cost_matrix(distances)

    s = routing.SolverSettings()
    utils.set_limits_for_quality(s, nodes)

    routing_solution = routing.Solve(d, s)
    final_cost = routing_solution.get_total_objective()
    vehicle_count = routing_solution.get_vehicle_count()
    cu_route = routing_solution.get_route()
    cu_status = routing_solution.get_status()

    # status returns integer instead of enum
    assert cu_status == 0
    # FIXME find better error rates
    assert (final_cost - ref_cost) / ref_cost < 0.2
    assert cu_route["route"].unique().count() == nodes
    assert cu_route["truck_id"].unique().count() == vehicle_count
