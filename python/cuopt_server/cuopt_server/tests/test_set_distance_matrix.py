# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient
from cuopt_server.utils.routing.solver import (
    _distance_tier_threshold_for_solver,
)
from cuopt_server.utils.routing.validation_distance_matrix import (
    validate_distance_matrix,
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
        "error_result": True,
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
        "error_result": True,
    }


def test_invalid_shape_set_distance_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["distance_matrix_data"] = {"data": {0: [[0, 10, 20], [10, 0, 15]]}}

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Distance matrix must be a square matrix",
        "error_result": True,
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
        "error_result": True,
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
        "error_result": True,
    }


def test_invalid_distance_matrix_requires_tiers(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    del data["fleet_data"]["vehicle_distance_tiers"]

    response_set = validate_only(data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": (
            "vehicle_distance_tiers must be set when distance matrix data is "
            "provided"
        ),
        "error_result": True,
    }


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
        "error_result": True,
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
        "error_result": True,
    }


def test_invalid_extra_arg_set_distance_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["distance_matrix_data"] = {
        "data": {0: [[0, 10, 20], [10, 0, 15], [20, 15, 0]]},
        "extra_arg": 1,
    }

    response_set = validate_only(data)

    assert response_set.status_code == 422
