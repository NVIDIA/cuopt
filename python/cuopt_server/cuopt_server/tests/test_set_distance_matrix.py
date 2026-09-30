# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient
from cuopt_server.utils.routing.conversion import (
    _distance_tier_threshold_for_solver,
)
from cuopt_server.utils.routing.data_definition import WaypointGraph
from cuopt_server.utils.routing.validation_distance_matrix import (
    validate_distance_matrix,
)
from cuopt_server.utils.routing.validation_fleet_data import (
    _validate_distance_tiers,
)
from cuopt_server.utils.routing.optimization_data_model import (
    OptimizationDataModel,
)

client = RequestClient()

# SET DISTANCE MATRIX TESTING

valid_data = {
    "cost_matrix_data": {"data": {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}},
    "distance_matrix_data": {
        "data": {0: [[0, 10, 20], [10, 0, 15], [20, 15, 0]]}
    },
    "fleet_data": {
        "vehicle_locations": [[0, 0]],
        "vehicle_types": [0],
        "vehicle_distance_tiers": [
            [{"threshold": None, "fixed_cost": 50, "cost_per_unit": 0}]
        ],
        "vehicle_max_distances": [100],
    },
    "task_data": {
        "task_locations": [1, 2],
    },
    "solver_config": {"time_limit": 0.1},
}


def validate_only(data):
    return client.post(
        "/cuopt/request",
        params={"validation_only": True},
        json=data,
    )


def test_valid_set_distance_matrix(cuoptproc):  # noqa
    response_set = validate_only(valid_data)

    assert response_set.status_code == 200


def test_null_distance_tier_threshold_converts_to_open_ended_value():
    assert (
        _distance_tier_threshold_for_solver(None) == np.finfo(np.float32).max
    )
    assert _distance_tier_threshold_for_solver(100.0) == 100.0


def test_invalid_empty_set_distance_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["distance_matrix_data"] = {"data": {}}

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Distance matrix cannot be null or empty",
        "error_result": False,
    }


def test_invalid_row_length_set_distance_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["distance_matrix_data"] = {
        "data": {0: [[0, 10, 20], [10, 0, 15], [20, 15]]}
    }

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All rows in the distance matrix must be of the same length",
        "error_result": False,
    }


def test_invalid_shape_set_distance_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["distance_matrix_data"] = {"data": {0: [[0, 10, 20], [10, 0, 15]]}}

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Distance matrix must be a square matrix",
        "error_result": False,
    }


def test_invalid_negative_values_set_distance_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["distance_matrix_data"] = {
        "data": {0: [[0, 10, 20], [10, 0, 15], [20, -15, 0]]}
    }

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All values in distance matrix must be >= 0",
        "error_result": False,
    }


def test_invalid_infinite_values_validate_distance_matrix():
    is_valid, msg = validate_distance_matrix(
        {0: [[0, 10, 20], [10, 0, float("inf")], [20, 15, 0]]},
        vehicle_distance_tiers=[
            [{"threshold": None, "fixed_cost": 50, "cost_per_unit": 0}]
        ],
    )

    assert is_valid is False
    assert msg == "All values in distance matrix must be finite"


def test_invalid_matrices_shape_set_distance_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["distance_matrix_data"] = {
        "data": {
            0: [[0, 10, 20], [10, 0, 15], [20, 15, 0]],
            1: [[0, 10], [10, 0]],
        }
    }

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Distance matrices for all vehicle types must be the same shape",
        "error_result": False,
    }


def test_distance_matrix_shape_must_match_cost_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["distance_matrix_data"] = {"data": {0: [[0, 10], [10, 0]]}}

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Distance matrix shape must match the cost matrix shape",
        "error_result": False,
    }


def test_distance_matrix_values_must_fit_float32():
    is_valid, msg = validate_distance_matrix(
        {0: [[0, 1e100], [1, 0]]},
        vehicle_distance_tiers=[
            [{"threshold": None, "fixed_cost": 50, "cost_per_unit": 0}]
        ],
    )

    assert is_valid is False
    assert (
        msg == "All values in distance matrix must be representable as float32"
    )


def test_distance_matrix_vehicle_type_must_fit_uint8():
    is_valid, msg = validate_distance_matrix(
        {256: [[0, 1], [1, 0]]},
        require_distance_tiers=False,
    )

    assert is_valid is False
    assert msg == "Matrix vehicle types must be integers within [0, 255]"


def test_distance_matrix_vehicle_type_must_have_cost_matrix():
    is_valid, msg = validate_distance_matrix(
        {1: [[0, 1], [1, 0]]},
        require_distance_tiers=False,
        comparison_matrix={0: np.zeros((2, 2), dtype=np.float32)},
    )

    assert is_valid is False
    assert msg == "Distance matrix shape must match the cost matrix shape"


def test_distance_tiers_require_strictly_increasing_thresholds():
    is_valid, msg = _validate_distance_tiers(
        [
            [
                {"threshold": 10, "fixed_cost": 1, "cost_per_unit": 0},
                {"threshold": 10, "fixed_cost": 0, "cost_per_unit": 1},
                {"threshold": None, "fixed_cost": 0, "cost_per_unit": 2},
            ]
        ]
    )

    assert is_valid is False
    assert msg == "Distance tier thresholds must be strictly increasing"


def test_distance_tier_thresholds_must_remain_distinct_as_float32():
    is_valid, msg = _validate_distance_tiers(
        [
            [
                {"threshold": 1.00000001, "fixed_cost": 0, "cost_per_unit": 1},
                {"threshold": 1.00000002, "fixed_cost": 0, "cost_per_unit": 1},
                {"threshold": None, "fixed_cost": 0, "cost_per_unit": 1},
            ]
        ]
    )

    assert is_valid is False
    assert msg == "Distance tier thresholds must be strictly increasing"


def test_explicit_float32_max_threshold_cannot_precede_open_ended_tier():
    is_valid, msg = _validate_distance_tiers(
        [
            [
                {
                    "threshold": np.finfo(np.float32).max,
                    "fixed_cost": 0,
                    "cost_per_unit": 1,
                },
                {"threshold": None, "fixed_cost": 0, "cost_per_unit": 1},
            ]
        ]
    )

    assert is_valid is False
    assert msg == "Distance tier thresholds must be strictly increasing"


def test_distance_tiers_require_one_final_open_ended_tier():
    is_valid, msg = _validate_distance_tiers(
        [
            [
                {"threshold": None, "fixed_cost": 1, "cost_per_unit": 0},
                {"threshold": None, "fixed_cost": 0, "cost_per_unit": 1},
            ]
        ]
    )

    assert is_valid is False
    assert msg == "The open-ended distance tier must be the final tier"


def test_distance_matrix_supports_max_distance_without_tiers(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    del data["fleet_data"]["vehicle_distance_tiers"]

    response_set = validate_only(data)

    assert response_set.status_code == 200


def test_vehicle_max_distance_must_fit_float32(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["fleet_data"]["vehicle_max_distances"] = [1e100]

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json()["error"] == (
        "Maximum distance any vehicle can travel must be representable as float32"
    )


def test_vehicle_type_must_fit_uint8(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["fleet_data"]["vehicle_types"] = [256]

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert (
        response_set.json()["error"] == "Vehicle types must be within [0, 255]"
    )


def test_update_distance_matrix_supports_max_distance_without_tiers():
    data_model = OptimizationDataModel()
    matrix = {0: [[0, 1], [1, 0]]}

    assert data_model.set_cost_matrix(matrix)[0]
    assert data_model.set_distance_matrix(matrix, None)[0]
    assert data_model.update_distance_matrix({0: [[0, 2], [2, 0]]})[0]


def test_distance_matrix_allows_cost_waypoint_graph():
    data_model = OptimizationDataModel()
    waypoint_graph = WaypointGraph(
        edges=[1, 0], offsets=[0, 1], weights=[1.0, 1.0]
    )

    assert data_model.set_cost_waypoint_graph({0: waypoint_graph})[0]
    assert data_model.set_distance_matrix({0: [[0, 1], [1, 0]]}, None)[0]


def test_validation_only_checks_distance_shape_after_waypoint_preparation(
    cuoptproc,  # noqa
):
    data = copy.deepcopy(valid_data)
    del data["cost_matrix_data"]
    data["cost_waypoint_graph_data"] = {
        "waypoint_graph": {
            0: {
                "edges": [1, 2, 0, 2, 0, 1],
                "offsets": [0, 2, 4, 6],
                "weights": [1, 1, 1, 1, 1, 1],
            }
        }
    }
    data["distance_matrix_data"] = {"data": {0: [[0, 1], [1, 0]]}}

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json()["error"] == (
        "Distance matrix shape must match the cost matrix shape"
    )


def test_default_vehicle_type_requires_zero_matrix_key():
    data_model = OptimizationDataModel()
    assert data_model.set_cost_matrix({1: [[0, 1], [1, 0]]})[0]

    is_valid = data_model.set_fleet_data(
        None,
        [[0, 0]],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    assert is_valid == (
        False,
        "Set vehicle types when using multiple matrices",
    )


def test_invalid_distance_tiers_require_distance_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    del data["distance_matrix_data"]
    del data["fleet_data"]["vehicle_max_distances"]

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": (
            "distance_matrix_data must be set when vehicle_distance_tiers is "
            "provided"
        ),
        "error_result": False,
    }


def test_invalid_vehicle_max_distances_require_distance_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    del data["distance_matrix_data"]
    del data["fleet_data"]["vehicle_distance_tiers"]

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": (
            "distance_matrix_data must be set when vehicle_max_distances is "
            "provided"
        ),
        "error_result": False,
    }


def test_invalid_extra_arg_set_distance_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["distance_matrix_data"] = {
        "data": {0: [[0, 10, 20], [10, 0, 15], [20, 15, 0]]},
        "extra_arg": 1,
    }

    response_set = validate_only(data)

    assert response_set.status_code == 422
