# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient

client = RequestClient()

# FLEET DATA TESTING

valid_data = {
    "cost_matrix_data": {
        "data": {
            0: [
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 1, 0],
            ]
        }
    },
    "travel_time_matrix_data": {
        "data": {
            0: [
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 1, 0],
            ]
        }
    },
    "fleet_data": {
        "vehicle_locations": [[1, 1], [2, 2], [3, 3], [4, 4]],
        "vehicle_time_windows": [[0, 10], [0, 10], [0, 10], [0, 10]],
    },
    "task_data": {
        "task_locations": [1],
        "demand": [[1], [1]],
        "task_time_windows": [[0, 10]],
        "service_times": [1],
    },
    "solver_config": {
        "time_limit": 0.5,
        "objectives": {
            "cost": 100,
            "travel_time": 200,
            "variance_route_size": 10,
            "variance_route_service_time": 50,
        },
        "config_file": "config.yaml",
    },
}


# Test valid set solver config with all fleet parameters
def test_valid_full_set_solver_config(cuoptproc):  # noqa
    response_set = client.post("/cuopt/request", json=valid_data)
    assert response_set.status_code == 200


# Test valid set solver config with minimal required parameters
def test_valid_minimal_set_solver_config(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["solver_config"] = {"time_limit": 0.1}

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 200


# Test invalid solver config values are in range
def test_invalid_values_set_solver_config(cuoptproc):  # noqa
    invalid_solver_config = {"time_limit": 0, "config_file": ""}

    # time limit must be greater than 0
    test_data = copy.deepcopy(valid_data)
    test_data["solver_config"]["time_limit"] = invalid_solver_config[
        "time_limit"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "SolverSettings time limit must be greater than 0",
        "error_result": True,
    }

    # config_file should be a valid file path
    test_data = copy.deepcopy(valid_data)
    test_data["solver_config"]["config_file"] = invalid_solver_config[
        "config_file"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "File path to save configuration should be valid and not empty",  # noqa
        "error_result": True,
    }


def test_ivalid_extra_arg_set_solver_config(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["solver_config"] = {
        "time_limit": 0.1,
        "extra_arg": 1,
    }

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 422
