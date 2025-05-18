# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.   # noqa
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

import time

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient

client = RequestClient()


def test_solver_logging(cuoptproc):  # noqa

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

    params = {"solver_logs": True}
    res = client.post("/cuopt/request", params=params, json=data, block=False)
    assert res.status_code == 200
    reqId = res.json()["reqId"]

    # Loop until the log is found
    cnt = 0
    while cnt < 60:
        res = client.get(
            f"/cuopt/log/{reqId}/", headers={"Accept": "application/json"}
        )
        if "error" not in res.json():
            break
        time.sleep(1)
        cnt += 1

    i = res.json()
    assert "log" in i
    assert "nbytes" in i
    assert isinstance(i["log"], list)
    assert len(i["log"]) > 0
    assert isinstance(i["nbytes"], int)
