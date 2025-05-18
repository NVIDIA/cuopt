# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

import os

import cuopt_mps_parser
import msgpack

from cuopt.linear_programming import solver_settings
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_INFEASIBILITY_DETECTION,
)

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import (
    RAPIDS_DATASET_ROOT_DIR,
    RequestClient,
)

client = RequestClient()


def test_warmstart(cuoptproc):  # noqa
    file_path = os.path.join(
        RAPIDS_DATASET_ROOT_DIR,
        "linear_programming/square41/square41.mps",
    )
    data_model_obj = cuopt_mps_parser.ParseMps(file_path)
    data = cuopt_mps_parser.toDict(data_model_obj, json=True)
    settings = solver_settings.SolverSettings()
    settings.set_optimality_tolerance(1e-4)
    settings.set_parameter(CUOPT_INFEASIBILITY_DETECTION, False)
    data["solver_config"] = settings.toDict()

    headers = {"CLIENT-VERSION": "custom"}

    res = client.post(
        "/cuopt/request",
        headers=headers,
        json=data,
        delete=False,
    )
    assert res.status_code == 200
    response = res.json()["response"]["solver_response"]
    assert response["status"] == 1
    solve_1_iter = response["solution"]["lp_statistics"]["nb_iterations"]

    settings.set_optimality_tolerance(1e-3)
    data["solver_config"] = settings.toDict()

    res = client.post(
        "/cuopt/request",
        headers=headers,
        json=data,
        delete=False,
    )
    assert res.status_code == 200
    response = res.json()["response"]["solver_response"]
    reqId = res.json()["reqId"]
    assert response["status"] == 1
    solve_2_iter = response["solution"]["lp_statistics"]["nb_iterations"]

    res = client.get(
        f"/cuopt/solution/{reqId}/warmstart",
    )
    assert res.status_code == 200
    res = msgpack.loads(res.content)
    settings.set_optimality_tolerance(1e-4)
    data["solver_config"] = settings.toDict()

    params = {"warmstartId": reqId}
    res = client.post(
        "/cuopt/request",
        headers=headers,
        params=params,
        json=data,
        delete=False,
    )
    assert res.status_code == 200
    response = res.json()["response"]["solver_response"]
    assert response["status"] == 1
    solve_3_iter = response["solution"]["lp_statistics"]["nb_iterations"]

    assert solve_3_iter + solve_2_iter == solve_1_iter
