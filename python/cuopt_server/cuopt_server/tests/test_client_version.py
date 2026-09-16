# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_server.utils.client_version import check_client_version


def test_check_client_accepts_lowercase_true(monkeypatch):
    monkeypatch.setenv("CUOPT_CHECK_CLIENT", "true")
    warnings = check_client_version("0.0")
    assert warnings
    assert "0.0" in warnings[0]
    assert "matching client" in warnings[0]


def test_check_client_disabled_when_false(monkeypatch):
    monkeypatch.setenv("CUOPT_CHECK_CLIENT", "False")
    assert check_client_version("0.0") == []
