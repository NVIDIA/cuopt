# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient

client = RequestClient()


def test_incumbent_callback(cuoptproc):  # noqa
    data = {
        "csr_constraint_matrix": {
            "offsets": [0, 2],
            "indices": [0, 1],
            "values": [1.0, 1.0],
        },
        "constraint_bounds": {"upper_bounds": [5000.0], "lower_bounds": [0.0]},
        "objective_data": {
            "coefficients": [1.2, 1.7],
            "scalability_factor": 1.0,
            "offset": 0.0,
        },
        "variable_bounds": {
            "upper_bounds": [3000.0, 5000.0],
            "lower_bounds": [0.0, 0.0],
        },
        "maximize": "True",
        "variable_names": ["x", "y"],
        "variable_types": ["I", "I"],
        "solver_config": {
            "time_limit": 30,
            "tolerances": {"optimality": 0.0001},
        },
    }

    params = {"incumbent_solutions": True}
    res = client.post("/cuopt/request", params=params, json=data, block=False)
    assert res.status_code == 200
    reqId = res.json()["reqId"]

    cnt = 0
    while cnt < 60:
        res = client.get(f"/cuopt/solution/{reqId}/incumbents")
        if res.json() != []:
            i = res.json()[0]
            assert "solution" in i
            assert isinstance(i["solution"], list)
            assert len(i["solution"]) > 0
            assert "cost" in i
            assert isinstance(i["cost"], float)
            break
        time.sleep(1)
        cnt += 1

    # Wait for sentinel
    saw_sentinel = False
    cnt = 0
    while cnt < 60:
        res = client.get(f"/cuopt/solution/{reqId}/incumbents").json()
        if len(res) == 1 and res[0] == {"solution": [], "cost": None}:
            saw_sentinel = True
            break
        time.sleep(1)
        cnt += 1
    assert saw_sentinel
