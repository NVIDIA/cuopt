# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

from cuopt_server.tests.utils.utils import RequestClient, cuoptproc  # noqa

client = RequestClient()


def test_health(cuoptproc):  # noqa
    # Normal health check
    response = client.get("/cuopt/health")
    assert response.status_code == 200

    # health check with root path
    response = client.get("/")
    assert response.status_code == 200


def test_readiness(cuoptproc):  # noqa
    response = client.get("/v2/health/ready")
    assert response.status_code == 200


def test_liveness(cuoptproc):  # noqa
    response = client.get("/v2/health/live")
    assert response.status_code == 200
