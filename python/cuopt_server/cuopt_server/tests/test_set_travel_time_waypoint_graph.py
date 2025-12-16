# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient

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
    "travel_time_waypoint_graph_data": {
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
        "task_time_windows": [[0, 10], [0, 10]],
        "service_times": [1, 1],
    },
    "fleet_data": {
        "vehicle_locations": [[0, 0]],
        "vehicle_time_windows": [[0, 10]],
        "vehicle_types": [0],
    },
    "solver_config": {"time_limit": 0.1},
}

client = RequestClient()


# Test valid travel time waypoint graph data
def test_valid_routes(cuoptproc):  # noqa
    response_set = client.post("/cuopt/request", json=valid_data)

    assert response_set.status_code == 200


# test that topology matches primary
def test_invalid_topology_routes(cuoptproc):  # noqa
    graph_data_mismatch_edges = {
        "waypoint_graph": {
            0: {
                "edges": [3, 0, 2, 3, 3],
                "offsets": [0, 1, 4, 5],
                "weights": [1.1, 1.3, 2.5, 4.5, 6.6, 7.0],
            }
        }
    }

    graph_data_mismatch_offsets = {
        "waypoint_graph": {
            0: {
                "edges": [3, 0, 2, 3, 3, 2],
                "offsets": [0, 1, 4],
                "weights": [1.1, 1.3, 2.5, 4.5, 6.6, 7.0],
            }
        }
    }

    test_data = copy.deepcopy(valid_data)
    test_data["travel_time_waypoint_graph_data"] = graph_data_mismatch_edges

    response_set_edges = client.post("/cuopt/request", json=test_data)

    assert response_set_edges.status_code == 400
    assert response_set_edges.json() == {
        "error": "Graph topology of primary and travel time waypoint graphs must match. Travel Time waypoint graph edges must match primary waypoint graph edges",  # noqa
        "error_result": True,
    }

    test_data["travel_time_waypoint_graph_data"] = graph_data_mismatch_offsets
    response_set_offsets = client.post(
        "/cuopt/request",
        json=test_data,
    )

    assert response_set_offsets.status_code == 400
    assert response_set_offsets.json() == {
        "error": "Graph topology of primary and travel time waypoint graphs must match. Travel Time waypoint graph offsets must match primary waypoint graph offsets",  # noqa
        "error_result": True,
    }


def test_invalid_extra_arg_routes(cuoptproc):  # noqa
    graph_data = {
        "edges": [3, 0, 2, 3, 3, 2],
        "offsets": [0, 1, 4, 5],
        "weights": [1.1, 1.3, 2.5, 4.5, 6.6, 7.0],
        "extra_arg": 1,
    }
    test_data = copy.deepcopy(valid_data)
    test_data["travel_time_waypoint_graph_data"]["waypoint_graph"] = {
        0: graph_data
    }

    response_set = client.post("/cuopt/request", json=test_data)

    assert response_set.status_code == 422
