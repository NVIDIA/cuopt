# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
