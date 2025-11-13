# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient, get_routes

client = RequestClient()


def test_initial_solutions(cuoptproc):  # noqa
    cost_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}

    # fleet data
    v_locations = [[0, 0], [0, 0]]

    # task data
    t_locations = [0, 1, 2]

    # submit a long running job
    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        task_locations=t_locations,
        delete=False,
    )
    assert res.status_code == 200
    assert "reqId" in res.json()
    reqId = res.json()["reqId"]
    print(reqId)
    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        task_locations=t_locations,
        initialId=[reqId, reqId, reqId, reqId, reqId],
    )

    assert res.status_code == 200
    assert "initial_solutions" in res.json()["response"]["solver_response"]
