# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
