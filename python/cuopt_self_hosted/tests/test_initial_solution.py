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

from cuopt_sh_client import CuOptServiceSelfHostClient


def test_initial_solution():
    port = os.environ.get("CUOPT_SERVER_PORT", 5000)

    client_cert = os.environ.get("CLIENT_CERT", "")
    use_https = client_cert != ""

    # Use initial solution
    data = {
        "cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
        "task_data": {"task_locations": [0, 1]},
        "fleet_data": {"vehicle_locations": [[0, 0], [0, 0]]},
        "initial_solution": [
            {
                "0": {
                    "task_id": ["Depot", "0", "1", "Depot"],
                    "type": ["Depot", "Delivery", "Delivery", "Depot"],
                }
            }
        ],
    }

    client = CuOptServiceSelfHostClient(
        port=port, use_https=use_https, self_signed_cert=client_cert
    )

    # Save this solution and use just the request id
    internal_initial_solution = client.get_optimized_routes(
        data, delete_solution=False
    )
    int_reqid = internal_initial_solution["reqId"]

    external_initial_solution = {
        "0": {
            "task_id": ["Depot", "0", "1", "Depot"],
            "type": ["Depot", "Delivery", "Delivery", "Depot"],
        }
    }
    # Upload a solution
    ext_reqid = client.upload_solution(external_initial_solution)["reqId"]

    solution = client.get_optimized_routes(
        data, initial_ids=[ext_reqid, int_reqid]
    )

    assert internal_initial_solution["response"]["solver_response"][
        "initial_solutions"
    ]
    assert solution["response"]["solver_response"]["initial_solutions"]

    client.delete(int_reqid)
    client.delete(ext_reqid)
