# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import (
    RequestClient,
    delete_request,
    get_routes,
    poll_request,
)

client = RequestClient()


def test_job_abort(cuoptproc):  # noqa
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

    solver_time_limit = 30
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
        result_timeout=0,
    )

    # Should have returned immediately with a request id
    assert res.status_code == 200
    result = res.json()
    assert "reqId" in result and "response" not in result

    # Delete job
    reqid = res.json()["reqId"]
    res = delete_request(client, reqid)
    assert res.status_code == 200
    assert res.json()["queued"] + res.json()["running"] == 1

    # Delete again, it's already aborted
    res = delete_request(client, reqid)
    assert res.status_code == 200
    assert res.json()["queued"] + res.json()["running"] == 0

    # Get the result, which should report the cancelled job.
    res = poll_request(client, reqid)
    assert res.status_code == 409
    assert reqid in res.json()["error"] and "cancelled" in res.json()["error"]

    # Delete again, job should not exist after the get
    res = delete_request(client, reqid)
    assert res.status_code == 200
    assert res.json()["queued"] + res.json()["running"] == 0
