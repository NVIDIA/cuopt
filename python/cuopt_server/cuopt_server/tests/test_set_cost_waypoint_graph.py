# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient

client = RequestClient()

valid_data = {
    "cost_waypoint_graph_data": {
        "waypoint_graph": {
            0: {
                "edges": [3, 0, 2, 3, 3, 2],
                "offsets": [0, 1, 4, 5, 6],
                "weights": [1.1, 1.3, 2.5, 4.5, 6.6, 7.0],
            }
        }
    },
    "task_data": {
        "task_locations": [1, 2],
    },
    "fleet_data": {"vehicle_locations": [[0, 0]], "vehicle_types": [0]},
    "solver_config": {"time_limit": 0.1},
}


# WAYPOINT GRAPH TESTING
def test_valid_full_set_waypoint_graph(cuoptproc):  # noqa
    response_set = client.post("/cuopt/request", json=valid_data)

    assert response_set.status_code == 200


def test_invalid_values_set_waypoint_graph(cuoptproc):  # noqa
    invalid_graph_data = {
        "edges": [3, 0, -2, 3, 3, 2],
        "offsets": [0, 1, 4, -5, 6],
        "weights": [1.1, 1.3, -2.5, 4.5, 6.6, 7.0],
    }

    test_data = copy.deepcopy(valid_data)
    test_data["cost_waypoint_graph_data"]["waypoint_graph"][0]["edges"] = (
        invalid_graph_data["edges"]
    )

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "edge values must be greater than or equal to 0",
        "error_result": True,
    }

    test_data = copy.deepcopy(valid_data)
    test_data["cost_waypoint_graph_data"]["waypoint_graph"][0]["offsets"] = (
        invalid_graph_data["offsets"]
    )

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "offset values must be greater than or equal to 0",
        "error_result": True,
    }

    test_data = copy.deepcopy(valid_data)
    test_data["cost_waypoint_graph_data"]["waypoint_graph"][0]["weights"] = (
        invalid_graph_data["weights"]
    )

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "weight values must be greater than or equal to 0",
        "error_result": True,
    }


def test_invalid_offsets_set_waypoint_graph(cuoptproc):  # noqa
    graph_data = {
        "edges": [3, 0, 2, 3, 3, 2],
        "offsets": [0, 1, 4, 5, 3, 5, 6],
        "weights": [1.1, 1.3, 2.5, 4.5, 6.6, 7.0],
    }
    data = copy.deepcopy(valid_data)
    data["cost_waypoint_graph_data"] = {"waypoint_graph": {0: graph_data}}

    response_set = client.post("/cuopt/request", json=data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Length of edges array must be greater than or equal to the length of the offsets array",  # noqa
        "error_result": True,
    }


def test_invalid_weights_set_waypoint_graph(cuoptproc):  # noqa
    graph_data = {
        "edges": [3, 0, 2, 3, 3, 2],
        "offsets": [0, 1, 4, 5],
        "weights": [1.1, 1.3, 2.5, 4.5, 6.6],
    }
    data = copy.deepcopy(valid_data)
    data["cost_waypoint_graph_data"] = {"waypoint_graph": {0: graph_data}}

    response_set = client.post("/cuopt/request", json=data)

    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Length of weights array must be equal to edges array",
        "error_result": True,
    }


def test_invalid_extra_arg_set_waypoint_graph(cuoptproc):  # noqa
    graph_data = {
        "edges": [3, 0, 2, 3, 3, 2],
        "offsets": [0, 1, 4, 5],
        "weights": [1.1, 1.3, 2.5, 4.5, 6.6, 7.0],
        "extra_arg": 1,
    }
    data = copy.deepcopy(valid_data)
    data["cost_waypoint_graph_data"] = {"waypoint_graph": {0: graph_data}}

    response_set = client.post("/cuopt/request", json=data)

    assert response_set.status_code == 422
