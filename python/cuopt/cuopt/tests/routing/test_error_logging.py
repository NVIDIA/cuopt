# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf

from cuopt import routing

cost_matrix = cudf.DataFrame(
    [
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
    ]
)
time_window = cudf.DataFrame(
    {
        "earliest": [0, 2, 1, 0, 8],
        "latest": [30, 10, 10, 10, 10],
        "service": [2, 2, 2, 2, 2],
    }
)
demand = cudf.Series([0, 1, 2, 1, 2])
capacity = cudf.Series([5, 5])


@pytest.mark.skip(reason="No error logging in new solver")
def test_time_window_constraints():
    data_model = routing.DataModel(cost_matrix.shape[0], 2)
    data_model.add_cost_matrix(cost_matrix)

    infeasible_tw = cudf.DataFrame(
        {
            "earliest": [0, 2, 1, 0, 7],
            "latest": [30, 10, 10, 10, 10],
            "service": [10, 15, 15, 15, 15],
        }
    )
    data_model.set_order_time_windows(
        infeasible_tw["earliest"], infeasible_tw["latest"]
    )
    data_model.set_order_service_times(infeasible_tw["service"])

    data_model.add_capacity_dimension("n_orders", demand, capacity)

    solver_settings = routing.SolverSettings()
    solver_settings.set_error_logging_mode(True)
    solution = routing.Solve(data_model, solver_settings)
    assert solution.get_status() != 0
    assert (
        solution.get_message()
        == "Infeasible Solve - Try relaxing Time Window constraints"
    )


@pytest.mark.skip(reason="No error logging in new solver")
def test_break_constraints():
    vehicle_num = len(capacity)
    data_model = routing.DataModel(cost_matrix.shape[0], vehicle_num)
    data_model.add_cost_matrix(cost_matrix)
    data_model.set_order_time_windows(
        time_window["earliest"], time_window["latest"]
    )
    data_model.set_order_service_times(time_window["service"])

    data_model.add_capacity_dimension("n_orders", demand, capacity)

    break_times = [[10, 45]]

    num_breaks = len(break_times)
    vehicle_breaks_earliest = np.zeros([vehicle_num, num_breaks])
    vehicle_breaks_latest = np.zeros([vehicle_num, num_breaks])
    vehicle_breaks_duration = np.zeros([vehicle_num, num_breaks])
    for b in range(num_breaks):
        break_begin = break_times[b][0]
        break_end = break_times[b][1]
        break_duration = break_end - break_begin
        vehicle_breaks_earliest[:, b] = [break_begin] * vehicle_num
        vehicle_breaks_latest[:, b] = [break_begin] * vehicle_num
        vehicle_breaks_duration[:, b] = [break_duration] * vehicle_num

    for b in range(num_breaks):
        data_model.add_break_dimension(
            cudf.Series(vehicle_breaks_earliest[:, b]),
            cudf.Series(vehicle_breaks_latest[:, b]),
            cudf.Series(vehicle_breaks_duration[:, b]),
        )
    solver_settings = routing.SolverSettings()
    solver_settings.set_error_logging_mode(True)
    solution = routing.Solve(data_model, solver_settings)
    assert solution.get_status() != 0
    assert (
        solution.get_message()
        == "Infeasible Solve - Try relaxing Break constraints"
    )
