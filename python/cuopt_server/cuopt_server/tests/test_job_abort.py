# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import (
    RAPIDS_DATASET_ROOT_DIR,
    RequestClient,
    delete_request,
    get_routes,
    poll_request,
)

client = RequestClient()


# Keep this one first because if we don't the restart of the
# solver from another test may cause this to result in a 200
# (job is aborted while still on the queue)
def test_abort_on_complete(cuoptproc):  # noqa
    lp_path = RAPIDS_DATASET_ROOT_DIR + "/cuopt_service_data/good_lp.json"

    with open(lp_path, "rb") as infile:
        data = infile.read()

    res = client.post("/cuopt/request", data=data)
    assert res.status_code == 200
    reqId = res.json()["reqId"]

    # No deletion reported because it's already completed
    res = delete_request(client, reqId).json()
    assert res["running"] == 0


def test_abort_of_running(cuoptproc):  # noqa
    cost_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}

    # fleet data
    v_locations = [[0, 0], [0, 0]]

    # task data
    t_locations = [0, 1, 2]

    solver_time_limit = 600

    # submit a long running job
    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        task_locations=t_locations,
        time_limit=solver_time_limit,
        result_timeout=0,
    )
    assert res.status_code == 200
    assert "reqId" in res.json()
    reqId = res.json()["reqId"]

    # abort the long running job
    cnt = 0
    while True:
        res = delete_request(client, reqId, running=True)
        if res.json()["running"] != 0 or cnt == 1000:
            break
        time.sleep(1)
        cnt += 1
    assert res.status_code == 200
    assert res.json()["running"] == 1

    # now a quick job should finish soon
    lp_path = RAPIDS_DATASET_ROOT_DIR + "/cuopt_service_data/good_lp.json"
    with open(lp_path, "rb") as infile:
        data = infile.read()

    then = time.time()
    res = client.post("/cuopt/request", data=data)
    assert res.status_code == 200
    assert time.time() - then < 60.0


def test_abort_queued_and_running_flags(cuoptproc):  # noqa
    cost_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}

    # fleet data
    v_locations = [[0, 0], [0, 0]]

    # task data
    t_locations = [0, 1, 2]

    solver_time_limit = 60

    ids = []
    for i in range(2):
        res = get_routes(
            client,
            cost_matrix=cost_matrix,
            vehicle_locations=v_locations,
            task_locations=t_locations,
            time_limit=solver_time_limit,
            result_timeout=0,
        )
        assert res.status_code == 200
        assert "reqId" in res.json()
        ids.append(res.json()["reqId"])

    time.sleep(3)

    res = delete_request(client, ids[0], queued=True, running=False)
    assert res.json()["cached"] == 0
    assert res.json()["queued"] == 0
    assert res.json()["running"] == 0

    res = delete_request(client, ids[1], queued=False, running=True)
    assert res.json()["cached"] == 0
    assert res.json()["queued"] == 0
    assert res.json()["running"] == 0

    res = delete_request(client, ids[1], queued=True, running=False)
    assert res.json()["cached"] == 0
    assert res.json()["queued"] == 1
    assert res.json()["running"] == 0

    res = delete_request(client, ids[0], queued=False, running=True)
    assert res.json()["cached"] == 0
    assert res.json()["queued"] == 0
    assert res.json()["running"] == 1


def test_abort_all(cuoptproc):  # noqa
    cost_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}

    # fleet data
    v_locations = [[0, 0], [0, 0]]

    # task data
    t_locations = [0, 1, 2]

    solver_time_limit = 30

    ids = []
    for i in range(20):
        res = get_routes(
            client,
            cost_matrix=cost_matrix,
            vehicle_locations=v_locations,
            task_locations=t_locations,
            time_limit=solver_time_limit,
            result_timeout=0,
        )
        assert res.status_code == 200
        assert "reqId" in res.json()
        ids.append(res.json()["reqId"])

    res = delete_request(client, "*", queued=True, running=True)
    assert res.json()["queued"] + res.json()["running"] == 20

    for i in ids:
        res = poll_request(client, i)
        assert res.status_code == 500
        assert i in res.json()["error"] and "aborted" in res.json()["error"]


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

    # Get the result, which should be a 500 and aborted message
    res = poll_request(client, reqid)
    assert res.status_code == 500
    assert reqid in res.json()["error"] and "aborted" in res.json()["error"]

    # Delete again, job should not exist after the get
    res = delete_request(client, reqid)
    assert res.status_code == 200
    assert res.json()["queued"] + res.json()["running"] == 0
