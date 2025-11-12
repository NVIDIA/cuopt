# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import (
    RequestClient,
    delete_request,
    get_routes,
)

client = RequestClient()


def validate_solver_sol(
    res,
    expected_status,
    expected_cost,
    expected_vehicle_count,
    expected_objective_values,
):
    assert res["status"] == expected_status
    assert res["solution_cost"] == expected_cost
    assert res["num_vehicles"] == expected_vehicle_count

    objective_values = res["objective_values"]
    for exp_type, exp_val in expected_objective_values.items():
        assert exp_val == objective_values[exp_type]


def test_job_cache(cuoptproc):  # noqa
    cost_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}
    time_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}

    # fleet data
    v_locations = [[0, 0], [0, 0]]
    v_capacities = [[2, 2], [4, 1]]
    v_time_windows = [[0, 10], [0, 10]]
    v_skip_first_trips = [False, False]
    v_drop_return_trips = [False, False]

    # task data
    t_locations = [0, 1, 2]
    t_demand = [[0, 1, 1], [0, 3, 1]]
    t_time_window = [[0, 10], [0, 4], [2, 4]]
    t_service_time = [0, 1, 1]

    solver_time_limit = 4
    vehicle_max_costs = [20, 20]
    vehicle_max_times = [10, 10]
    objectives = {"cost": 1}

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        travel_time_matrix=time_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
        vehicle_max_times=vehicle_max_times,
        objectives=objectives,
        cache=True,
    )

    # Should have cached but not solved
    assert res.status_code == 200
    result = res.json()
    assert "reqId" in result and "response" not in result

    # Reference the cached data and run
    cache_id = res.json()["reqId"]
    res = get_routes(client, reqId=cache_id)
    assert res.status_code == 200
    result = res.json()
    assert (
        "reqId" in result
        and "response" in result
        and result["reqId"] != cache_id
    )
    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_cost=3.0,
        expected_vehicle_count=1,
        expected_objective_values={"cost": 3.0},
    )

    # Delete cached data
    res = delete_request(client, cache_id)
    assert res.status_code == 200
    assert res.json()["cached"] == 1

    # Delete again, id is not present
    res = delete_request(client, cache_id)
    assert res.status_code == 200
    assert res.json()["cached"] == 0
