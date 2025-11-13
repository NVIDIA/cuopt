# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_server.tests.utils.utils import (  # noqa
    RequestClient,
    cuoptproc,
    get_routes,
)

client = RequestClient()


def test_multi_cost_matrix(cuoptproc):  # noqa
    bikes_type = 1
    car_type = 2
    bikes_cost = [[0, 4, 4], [4, 0, 4], [4, 4, 0]]
    bikes_time = [[0, 50, 50], [50, 0, 50], [50, 50, 0]]
    car_cost = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    car_time = [[0, 10, 10], [10, 0, 10], [10, 10, 0]]
    v_types = [bikes_type, car_type]
    cost_matrix = {bikes_type: bikes_cost, car_type: car_cost}
    travel_time_matrix = {bikes_type: bikes_time, car_type: car_time}
    t_locations = [0, 1, 2]
    v_locations = [[0, 0], [0, 0]]

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        travel_time_matrix=travel_time_matrix,
        vehicle_locations=v_locations,
        vehicle_types=v_types,
        min_vehicles=2,
        task_locations=t_locations,
        time_limit=4,
    )
    assert res.status_code == 200


def test_multi_waypoint_graph(cuoptproc):  # noqa
    bikes_type = 1
    car_type = 2

    bikes_cost_graph = {
        "edges": [1, 2, 0, 2, 0, 1],
        "offsets": [0, 2, 4, 6],
        "weights": [4, 4, 4, 4, 4, 4],
    }
    bikes_time_graph = {
        "edges": [1, 2, 0, 2, 0, 1],
        "offsets": [0, 2, 4, 6],
        "weights": [50, 50, 50, 50, 50, 50],
    }

    car_cost_graph = {
        "edges": [1, 0, 2, 1],
        "offsets": [0, 1, 3, 4],
        "weights": [1, 1, 1, 1],
    }
    car_time_graph = {
        "edges": [1, 0, 2, 1],
        "offsets": [0, 1, 3, 4],
        "weights": [10, 10, 10, 10],
    }

    wp_graph = {bikes_type: bikes_cost_graph, car_type: car_cost_graph}
    travel_time_wp_graph = {
        bikes_type: bikes_time_graph,
        car_type: car_time_graph,
    }

    v_types = [bikes_type, car_type]
    v_ids = ["Bike", "Car"]

    v_locations = [[0, 0], [0, 0]]

    t_locations = [0, 1, 2]
    res = get_routes(
        client,
        cost_waypoint_graph=wp_graph,
        travel_time_waypoint_graph=travel_time_wp_graph,
        vehicle_locations=v_locations,
        vehicle_ids=v_ids,
        vehicle_types=v_types,
        min_vehicles=2,
        task_locations=t_locations,
        time_limit=4,
    )
    assert res.status_code == 200

    assert "solver_response" in res.json()["response"].keys()

    # solution = res.json()["response"]["solver_response"]
    # FIXME: Determinism PR
    # assert res.status_code == 200
    # assert solution["solution_cost"] == 4.0
    # assert list(solution["vehicle_data"].keys()) == v_ids
    # assert solution["vehicle_data"]["Car"]["route"] == [0, 1, 2, 1, 0]
