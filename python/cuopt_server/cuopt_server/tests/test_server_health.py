# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_server.tests.utils.utils import RequestClient, cuoptproc  # noqa


def test_server_health(cuoptproc):  # noqa
    assert RequestClient().get("/cuopt/health").status_code == 200
