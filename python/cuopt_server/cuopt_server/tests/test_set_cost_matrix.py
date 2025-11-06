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

import copy

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient

client = RequestClient()

# SET COST MATRIX TESTING

valid_data = {
    "cost_matrix_data": {"data": {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}},
    "task_data": {
        "task_locations": [1, 2],
    },
    "fleet_data": {"vehicle_locations": [[0, 0]], "vehicle_types": [0]},
    "solver_config": {"time_limit": 0.1},
}


def test_valid_set_cost_matrix(cuoptproc):  # noqa
    response_set = client.post("/cuopt/request", json=valid_data)

    assert response_set.status_code == 200


# All cost matrix rows must be the same length
def test_invalid_row_length_cost_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["cost_matrix_data"] = {"data": {0: [[0, 1, 1], [1, 0, 1], [1, 1]]}}

    response_set = client.post("/cuopt/request", json=data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All rows in the cost matrix must be of the same length",
        "error_result": True,
    }


# Cost matrix must be a square matrix
def test_invalid_shape_set_cost_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["cost_matrix_data"] = {"data": {0: [[0, 1, 1], [1, 0, 1]]}}

    response_set = client.post("/cuopt/request", json=data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Cost matrix must be a square matrix",
        "error_result": True,
    }


# All cost matrix values must be greater than or equal to 0
def test_invalid_values_set_cost_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["cost_matrix_data"] = {
        "data": {0: [[0, 1, 1], [1, 0, 1], [1, -1, 0]]}
    }

    response_set = client.post("/cuopt/request", json=data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All values in cost matrix must be >= 0",
        "error_result": True,
    }


# Cost matrices for multiple vehicle types must have the same shape
def test_invalid_matrices_shape_set_cost_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["cost_matrix_data"] = {
        "data": {
            0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            1: [[0, 1], [1, 0]],
        }
    }

    response_set = client.post("/cuopt/request", json=data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Matrices for all vehicle types must be the same shape",
        "error_result": True,
    }


def test_valid_matrices_shape_set_cost_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["cost_matrix_data"] = {
        "data": {
            0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            1: [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        }
    }
    data["fleet_data"]["vehicle_types"] = [0]

    response_set = client.post("/cuopt/request", json=data)

    assert response_set.status_code == 200


def test_invalid_extra_arg_set_cost_matrix(cuoptproc):  # noqa
    data = copy.deepcopy(valid_data)
    data["cost_matrix_data"] = {
        "data": {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]},
        "extra_arg": 1,
    }

    response_set = client.post("/cuopt/request", json=data)

    assert response_set.status_code == 422
