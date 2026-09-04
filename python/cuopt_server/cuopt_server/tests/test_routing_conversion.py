# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_server.utils.routing import conversion
from cuopt_server.utils.routing.data_definition import (
    CostMatrices,
    FleetData,
    SolverSettingsConfig,
    TaskData,
)


def test_pydantic_request_converts_and_prepares_cost_matrix():
    optimization_data = conversion.populate_optimization_data(
        cost_matrix_data=CostMatrices(data={0: [[0, 1], [1, 0]]}),
        fleet_data=FleetData(vehicle_locations=[[0, 0]]),
        task_data=TaskData(task_locations=[1]),
        solver_config=SolverSettingsConfig(time_limit=1),
    )

    prepared, cost_matrix, travel_time_matrix, waypoint_graph = (
        conversion.prep_optimization_data(optimization_data)
    )

    assert prepared is optimization_data
    assert list(cost_matrix) == [0]
    assert cost_matrix[0].shape == (2, 2)
    assert travel_time_matrix is None
    assert waypoint_graph == {}


def test_default_solver_time_limit():
    solver_config = SolverSettingsConfig()
    optimization_data = conversion.populate_optimization_data(
        cost_matrix_data=CostMatrices(data={0: [[0, 1], [1, 0]]}),
        fleet_data=FleetData(vehicle_locations=[[0, 0]]),
        task_data=TaskData(task_locations=[1]),
        solver_config=solver_config,
    )

    assert solver_config.time_limit == 10 + 1 / 6
    assert optimization_data.solver_config["time_limit"] == 10 + 1 / 6
