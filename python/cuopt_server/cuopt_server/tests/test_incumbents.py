# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient

client = RequestClient()


def _run_incumbent_callback(cuoptproc, include_set_callback):  # noqa
    data = {
        "csr_constraint_matrix": {
            "offsets": [0, 3, 6, 9],
            "indices": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "values": [2.0, 1.0, 3.0, 4.0, 5.0, 1.0, 1.0, 2.0, 2.0],
        },
        "constraint_bounds": {
            "upper_bounds": [9000.0, 12000.0, 7000.0],
            "lower_bounds": [0.0, 0.0, 0.0],
        },
        "objective_data": {
            "coefficients": [3.1, 2.7, 1.9],
            "scalability_factor": 1.0,
            "offset": 0.0,
        },
        "variable_bounds": {
            "upper_bounds": [4000.0, 5000.0, 3500.0],
            "lower_bounds": [0.0, 0.0, 0.0],
        },
        "maximize": "True",
        "variable_names": ["x", "y", "z"],
        "variable_types": ["I", "I", "I"],
        "solver_config": {
            "time_limit": 30,
            "tolerances": {"optimality": 0.0001},
        },
    }

    params = {
        "incumbent_solutions": True,
        "incumbent_set_solutions": include_set_callback,
    }
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
            assert "bound" in i
            assert isinstance(i["bound"], float)
            break
        time.sleep(1)
        cnt += 1

    # Wait for sentinel
    saw_sentinel = False
    cnt = 0
    while cnt < 60:
        res = client.get(f"/cuopt/solution/{reqId}/incumbents").json()
        if len(res) == 1 and res[0] == {
            "solution": [],
            "cost": None,
            "bound": None,
        }:
            saw_sentinel = True
            break
        time.sleep(1)
        cnt += 1
    assert saw_sentinel


def test_incumbent_callback_get_only(cuoptproc):  # noqa
    _run_incumbent_callback(cuoptproc, include_set_callback=False)


def test_incumbent_callback_get_set(cuoptproc):  # noqa
    _run_incumbent_callback(cuoptproc, include_set_callback=True)
