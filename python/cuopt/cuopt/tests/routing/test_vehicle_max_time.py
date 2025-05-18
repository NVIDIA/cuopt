# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

import numpy as np

import cudf

from cuopt import routing
from cuopt.routing.vehicle_routing_wrapper import ErrorStatus


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
