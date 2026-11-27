# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from fastapi import HTTPException

from cuopt_server.utils.routing import conversion
from cuopt_server.utils.routing.data_definition import (
    CostMatrices,
    DistanceMatrices,
    FleetData,
    SolverSettingsConfig,
    TaskData,
    WaypointGraphData,
)
from cuopt_server.utils.routing.host_optimization_data_model import (
    HostOptimizationDataModel,
)
from cuopt_server.utils.routing.optimization_data_model import (
    OptimizationDataModel,
)


@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {0: []},
        {0: [1, 2]},
        {0: [[[1], [2]], [[3], [4]]]},
        {0: [[0, 2 + 4j], [3, 0]]},
        {0: [[0, 1], [1]]},
        {0: [[0, 1, 2], [1, 0, 3]]},
        {0: [[0, -1], [1, 0]]},
        {0: [[0, np.nan], [1, 0]]},
        {0: [[0, np.inf], [1, 0]]},
        {256: [[0]]},
        {0: [[0, float(np.finfo(np.float32).max) * 2.0], [1, 0]]},
    ],
)
def test_invalid_distance_inputs(data):
    assert not HostOptimizationDataModel().set_distance_matrix(data)[0]


@pytest.mark.parametrize(
    "model_class", [HostOptimizationDataModel, OptimizationDataModel]
)
def test_invalid_rank_leaves_registered_distances_intact(model_class):
    model = model_class()
    assert model.set_distance_matrix({0: [[0, 5], [3, 0]]})[0]
    for matrix in ([1, 2], [[[1], [2]], [[3], [4]]]):
        assert not model.set_distance_matrix({0: matrix})[0]
    assert model.get_distance_matrix() == {0: [[0, 5], [3, 0]]}


def test_distance_requires_matching_cost_matrix():
    model = HostOptimizationDataModel()
    assert model.set_cost_matrix({0: [[0, 3], [5, 0]]})[0]
    assert not model.set_distance_matrix({1: [[0, 1], [1, 0]]})[0]
    assert not model.set_distance_matrix({0: [[0]]})[0]
    assert model.set_distance_matrix({0: [[0, 1e30], [1, 0]]})[0]


def test_host_conversion_preserves_distance_matrix(monkeypatch):
    import cuopt_server.utils.routing.optimization_data_model as model_module

    def fail(*args, **kwargs):
        raise AssertionError(
            "host distance conversion allocated a cudf object"
        )

    monkeypatch.setattr(model_module.cudf, "DataFrame", fail)
    monkeypatch.setattr(model_module.cudf, "Series", fail)
    optimization = conversion.populate_optimization_data(
        cost_matrix_data=CostMatrices(data={0: [[0, 3], [5, 0]]}),
        distance_matrix_data=DistanceMatrices(data={0: [[0, 100], [200, 0]]}),
        fleet_data=FleetData(vehicle_locations=[[0, 0]]),
        task_data=TaskData(task_locations=[1]),
        solver_config=SolverSettingsConfig(time_limit=0.1),
    )
    prepared, costs, times, _ = conversion.prep_optimization_data(optimization)
    calls = []
    add_distance = conversion.routing.DataModel.add_distance_matrix

    def record_distance(model, matrix, vehicle_type):
        calls.append((matrix, vehicle_type))
        return add_distance(model, matrix, vehicle_type)

    monkeypatch.setattr(
        conversion.routing.DataModel, "add_distance_matrix", record_distance
    )
    conversion.create_data_model(
        prepared, cost_matrix=costs, travel_time_matrix=times
    )
    [(distance, vehicle_type)] = calls
    np.testing.assert_array_equal(distance, [[0, 100], [200, 0]])
    assert vehicle_type == 0
    assert np.asarray(distance).dtype == np.float32
    assert optimization.get_distance_matrix() == {0: [[0, 100], [200, 0]]}


@pytest.mark.parametrize(
    "model_class", [HostOptimizationDataModel, OptimizationDataModel]
)
def test_distance_update_behavior(model_class):
    model = model_class()
    assert model.set_cost_matrix({0: [[0, 3], [5, 0]]})[0]
    assert model.set_distance_matrix({0: [[0, 100], [200, 0]]})[0]
    if model_class is HostOptimizationDataModel:
        with pytest.raises(NotImplementedError):
            model.update_distance_matrix({0: [[0, 2], [2, 0]]})
    else:
        assert model.update_distance_matrix({0: [[0, 2], [2, 0]]})[0]
        assert model.get_distance_matrix() == {0: [[0, 2], [2, 0]]}


def test_distance_shape_rechecked_after_cost_preparation():
    optimization = conversion.populate_optimization_data(
        cost_waypoint_graph_data=WaypointGraphData(
            waypoint_graph={
                0: {
                    "edges": [1, 2, 0, 2, 0, 1],
                    "offsets": [0, 2, 4, 6],
                    "weights": [3, 4, 5, 6, 7, 8],
                }
            }
        ),
        distance_matrix_data=DistanceMatrices(data={0: [[0]]}),
        fleet_data=FleetData(vehicle_locations=[[0, 0]]),
        task_data=TaskData(task_locations=[1]),
        solver_config=SolverSettingsConfig(time_limit=0.1),
    )
    with pytest.raises(HTTPException, match="shape must match"):
        conversion.prep_optimization_data(optimization)


def test_update_distance_without_cost_preserves_registered_shape():
    model = OptimizationDataModel()
    assert model.set_distance_matrix({0: [[0, 3], [5, 0]]})[0]
    assert not model.update_distance_matrix({1: [[0]]})[0]
    assert not model.update_distance_matrix({0: [[0]]})[0]
    assert model.get_distance_matrix() == {0: [[0, 3], [5, 0]]}
    assert model.update_distance_matrix({0: [[0, 4], [6, 0]]})[0]


def test_partial_fleet_update_preserves_registered_vehicle_types():
    model = OptimizationDataModel()
    assert model.set_cost_matrix({1: [[0, 3], [5, 0]]})[0]
    assert model.set_distance_matrix({1: [[0, 9], [7, 0]]})[0]
    fleet = FleetData(vehicle_locations=[[0, 0]], vehicle_types=[1])
    assert model.set_fleet_data(**fleet.model_dump())[0]

    update = dict.fromkeys(fleet.model_dump())
    update["vehicle_max_times"] = [10]
    assert model.update_fleet_data(**update)[0]
    assert model.get_fleet_data()["vehicle_types"] == [1]
    assert model.get_fleet_data()["vehicle_max_times"] == [10]

    update["vehicle_types"] = [0]
    assert not model.update_fleet_data(**update)[0]
    assert model.get_fleet_data()["vehicle_types"] == [1]


@pytest.mark.parametrize(
    "matrix_type, fleet_types", [(0, None), (1, [1]), (1, None)]
)
def test_legacy_validation_only_checks_default_distance_type(
    matrix_type, fleet_types
):
    from cuopt_server.utils.deprecated.solver import (
        solve_optimized_routes_sync,
    )

    request = dict(
        cost_matrix_data=CostMatrices(data={matrix_type: [[0, 3], [5, 0]]}),
        distance_matrix_data=DistanceMatrices(
            data={matrix_type: [[0, 9], [7, 0]]}
        ),
        fleet_data=FleetData(
            vehicle_locations=[[0, 0]], vehicle_types=fleet_types
        ),
        task_data=TaskData(task_locations=[1]),
        validation_only=True,
    )
    if fleet_types is None and matrix_type != 0:
        with pytest.raises(
            HTTPException, match="default vehicle type 0"
        ) as exc:
            solve_optimized_routes_sync(**request)
        assert exc.value.status_code == 400
    else:
        response, _, _ = solve_optimized_routes_sync(**request)
        assert response["response"]["solver_response"]["status"] == 0


def test_json_request_without_distance_matrix_remains_valid(monkeypatch):
    from cuopt_server.utils.routing.data_definition import OptimizedRoutingData

    request = OptimizedRoutingData.model_validate(
        {
            "cost_matrix_data": {"data": {"0": [[0, 3], [5, 0]]}},
            "fleet_data": {"vehicle_locations": [[0, 0]]},
            "task_data": {"task_locations": [1]},
            "solver_config": {"time_limit": 0.1},
        }
    )
    optimization = conversion.populate_optimization_data(**dict(request))
    prepared, costs, times, _ = conversion.prep_optimization_data(optimization)

    def unexpected(*args, **kwargs):
        pytest.fail("An omitted distance matrix must not be registered")

    monkeypatch.setattr(
        conversion.routing.DataModel, "add_distance_matrix", unexpected
    )
    conversion.create_data_model(
        prepared, cost_matrix=costs, travel_time_matrix=times
    )


def test_json_distance_request_preserves_vehicle_type_keys():
    from cuopt_server.utils.routing.data_definition import OptimizedRoutingData

    request = OptimizedRoutingData.model_validate(
        {
            "cost_matrix_data": {"data": {"0": [[0, 3], [5, 0]]}},
            "distance_matrix_data": {"data": {"0": [[0, 100], [200, 0]]}},
            "fleet_data": {"vehicle_locations": [[0, 0]]},
            "task_data": {"task_locations": [1]},
            "solver_config": {"time_limit": 0.1},
        }
    )
    assert list(request.distance_matrix_data.data) == [0]
    optimization = conversion.populate_optimization_data(**dict(request))
    assert optimization.get_distance_matrix() == {0: [[0, 100], [200, 0]]}
