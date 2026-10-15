# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from cuopt.grpc.routing.grpc_client import problem_summary

from cuopt_server.utils.routing import conversion
from cuopt_server.utils.utils import build_routing_datamodel_from_json
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


def _dense_request():
    return {
        "cost_matrix_data": CostMatrices(data={0: [[0, 1], [1, 0]]}),
        "fleet_data": FleetData(vehicle_locations=[[0, 0]]),
        "task_data": TaskData(task_locations=[1]),
        "solver_config": SolverSettingsConfig(time_limit=1),
    }


def test_host_conversion_keeps_dense_request_off_gpu(monkeypatch):
    import cuopt_server.utils.routing.optimization_data_model as model_module

    def fail(*args, **kwargs):
        raise AssertionError("dense host conversion allocated a cudf object")

    monkeypatch.setattr(model_module.cudf, "Series", fail)
    monkeypatch.setattr(model_module.cudf, "DataFrame", fail)

    optimization_data = conversion.populate_optimization_data(
        **_dense_request()
    )
    prepared, cost_matrix, travel_time_matrix, _ = (
        conversion.prep_optimization_data(optimization_data)
    )
    _, data_model = conversion.create_data_model(
        prepared,
        cost_matrix=cost_matrix,
        travel_time_matrix=travel_time_matrix,
    )

    assert type(cost_matrix[0]).__module__.split(".", 1)[0] == "pandas"

    stored_cost, vehicle_type = data_model._recorded("add_cost_matrix")[0]
    assert vehicle_type == 0
    assert isinstance(stored_cost, np.ndarray)
    assert stored_cost.shape == (2, 2)
    np.testing.assert_array_equal(stored_cost, cost_matrix[0].to_numpy())
    assert all(
        type(arg).__module__.split(".", 1)[0] != "cudf"
        for _, args, _ in data_model._calls
        for arg in args
    )
    summary = problem_summary(data_model)
    assert summary["num_locations"] == 2
    assert summary["fleet_size"] == 1
    assert summary["num_orders"] == 1
    assert summary["cost_matrices"] == 1


def test_build_routing_datamodel_from_json_accepts_dict():
    data_model, solver_settings = build_routing_datamodel_from_json(
        {
            "cost_matrix_data": {"data": {0: [[0, 1], [1, 0]]}},
            "fleet_data": {"vehicle_locations": [[0, 0]]},
            "task_data": {"task_locations": [1]},
            "solver_config": {"time_limit": 1},
        }
    )
    summary = problem_summary(data_model)
    assert summary["num_locations"] == 2
    assert summary["fleet_size"] == 1
    assert summary["num_orders"] == 1
    assert summary["cost_matrices"] == 1
    assert solver_settings.get_time_limit() == 1


def test_host_optimization_model_updates_are_unimplemented():
    from cuopt_server.utils.routing.host_optimization_data_model import (
        HostOptimizationDataModel,
    )

    model = HostOptimizationDataModel()
    for name in (
        "update_cost_matrix",
        "update_travel_time_matrix",
        "update_fleet_data",
        "update_task_data",
        "update_solver_config",
    ):
        with pytest.raises(NotImplementedError):
            getattr(model, name)()
