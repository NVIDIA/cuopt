# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from fastapi import HTTPException

from cuopt_server import webserver
from cuopt_server.utils import settings


@pytest.fixture(autouse=True)
def restore_data_dir():
    original_data_dir = settings.get_data_dir()
    try:
        yield
    finally:
        settings.set_data_dir(original_data_dir)


def test_validate_file_path_returns_file_if_relative_path_stays_in_data_dir(
    tmp_path,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "input.json"
    data_file.write_text("{}", encoding="utf-8")
    settings.set_data_dir(str(data_dir))

    assert webserver.validate_file_path("input.json") == str(data_file)


def test_validate_file_path_rejects_absolute_path(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside_file = tmp_path / "input.json"
    outside_file.write_text("{}", encoding="utf-8")
    settings.set_data_dir(str(data_dir))

    with pytest.raises(HTTPException) as exc_info:
        webserver.validate_file_path(str(outside_file))

    assert exc_info.value.status_code == 400
    assert "relative to CUOPT_DATA_DIR" in exc_info.value.detail


def test_validate_file_path_rejects_parent_directory_escape(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside_file = tmp_path / "input.json"
    outside_file.write_text("{}", encoding="utf-8")
    settings.set_data_dir(str(data_dir))

    with pytest.raises(HTTPException) as exc_info:
        webserver.validate_file_path("../input.json")

    assert exc_info.value.status_code == 400
    assert "stay inside CUOPT_DATA_DIR" in exc_info.value.detail


def test_validate_file_path_rejects_symlink_escape(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside_file = tmp_path / "input.json"
    outside_file.write_text("{}", encoding="utf-8")
    os.symlink(outside_file, data_dir / "linked-input.json")
    settings.set_data_dir(str(data_dir))

    with pytest.raises(HTTPException) as exc_info:
        webserver.validate_file_path("linked-input.json")

    assert exc_info.value.status_code == 400
    assert "stay inside CUOPT_DATA_DIR" in exc_info.value.detail
