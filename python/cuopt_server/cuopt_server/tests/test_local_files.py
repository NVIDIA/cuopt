# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pickle
import stat

import pytest
from fastapi import HTTPException

from cuopt_server.utils import settings
from cuopt_server.utils.local_files import (
    decode_file_bytes,
    file_result_message,
    get_output_name,
    load_optimization_file,
    result_meets_threshold,
    validate_file_path,
    write_result_file,
)


@pytest.fixture(autouse=True)
def restore_data_dir():
    original_data_dir = settings.get_data_dir()
    try:
        yield
    finally:
        settings.set_data_dir(original_data_dir)


def test_get_output_name_empty_without_result_dir():
    assert get_output_name("", "problem.json", "out.json") == ""


def test_get_output_name_uses_requested_file(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    assert get_output_name(str(result_dir), "problem.json", "out.bin") == (
        "out.bin"
    )


def test_get_output_name_rejects_result_path_escape(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    name = get_output_name(str(result_dir), "problem.json", "../escape.bin")
    assert name == "problem.json.result"


def test_get_output_name_from_data_file(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    assert get_output_name(str(result_dir), "nested/problem.json", "") == (
        "problem.json.result"
    )


def test_validate_file_path_returns_file_in_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "input.json"
    data_file.write_text("{}", encoding="utf-8")
    settings.set_data_dir(str(data_dir))

    assert validate_file_path("input.json") == str(data_file)


def test_validate_file_path_rejects_unset_data_dir():
    settings.set_data_dir("")

    with pytest.raises(HTTPException) as exc_info:
        validate_file_path("input.json")

    assert exc_info.value.status_code == 400
    assert "cuopt data directory not set" in exc_info.value.detail


def test_validate_file_path_rejects_absolute_path(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside_file = tmp_path / "input.json"
    outside_file.write_text("{}", encoding="utf-8")
    settings.set_data_dir(str(data_dir))

    with pytest.raises(HTTPException) as exc_info:
        validate_file_path(str(outside_file))

    assert exc_info.value.status_code == 400
    assert "relative to CUOPT_DATA_DIR" in exc_info.value.detail


def test_validate_file_path_rejects_parent_directory_escape(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside_file = tmp_path / "input.json"
    outside_file.write_text("{}", encoding="utf-8")
    settings.set_data_dir(str(data_dir))

    with pytest.raises(HTTPException) as exc_info:
        validate_file_path("../input.json")

    assert exc_info.value.status_code == 400
    assert "stay inside CUOPT_DATA_DIR" in exc_info.value.detail


def test_validate_file_path_rejects_non_regular_file(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "input").mkdir()
    settings.set_data_dir(str(data_dir))

    with pytest.raises(HTTPException) as exc_info:
        validate_file_path("input")

    assert exc_info.value.status_code == 400
    assert "not a regular file" in exc_info.value.detail


def test_validate_file_path_rejects_symlink_escape(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside_file = tmp_path / "input.json"
    outside_file.write_text("{}", encoding="utf-8")
    os.symlink(outside_file, data_dir / "linked-input.json")
    settings.set_data_dir(str(data_dir))

    with pytest.raises(HTTPException) as exc_info:
        validate_file_path("linked-input.json")

    assert exc_info.value.status_code == 400
    assert "stay inside CUOPT_DATA_DIR" in exc_info.value.detail


def test_result_meets_threshold():
    assert result_meets_threshold("out", "/tmp", 250000, 250) is True
    assert result_meets_threshold("out", "/tmp", 249999, 250) is False
    assert result_meets_threshold("out", "/tmp", 1, 0) is True
    assert result_meets_threshold("", "/tmp", 10**9, 0) is False
    assert result_meets_threshold("out", "", 10**9, 0) is False


def test_write_result_file_and_mode(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    payload = b'{"ok": true}'

    path = write_result_file(str(result_dir), "sol.json", payload, mode=0o600)

    assert path == str(result_dir / "sol.json")
    assert (result_dir / "sol.json").read_bytes() == payload
    assert stat.S_IMODE((result_dir / "sol.json").stat().st_mode) == 0o600


def test_file_result_message_includes_notes_and_warnings():
    assert file_result_message("sol.json") == {"result_file": "sol.json"}
    assert file_result_message("sol.json", warnings=["w"], notes=["n"]) == {
        "result_file": "sol.json",
        "warnings": ["w"],
        "notes": ["n"],
    }


def test_load_optimization_file_json(tmp_path):
    problem = {"csr_constraint_matrix": {"offsets": [0, 1]}}
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(problem), encoding="utf-8")

    assert load_optimization_file(str(path)) == problem


def test_load_optimization_file_without_extension(tmp_path):
    problem = {"task_data": {"task_locations": [1]}}
    path = tmp_path / "problem"
    path.write_text(json.dumps(problem), encoding="utf-8")

    assert load_optimization_file(str(path)) == problem


def test_load_pickle_appends_deprecation_warning(tmp_path):
    problem = {"csr_constraint_matrix": {"offsets": [0, 1]}}
    path = tmp_path / "problem.pickle"
    path.write_bytes(pickle.dumps(problem))
    warnings = []

    assert load_optimization_file(str(path), warnings) == problem
    assert warnings == [
        "Pickle data format is deprecated. Use zlib, msgpack, or plain JSON"
    ]


def test_load_pickle_forbidden_class(tmp_path):
    path = tmp_path / "bad.pickle"
    path.write_bytes(pickle.dumps({"obj": object()}))

    with pytest.raises(HTTPException) as exc_info:
        load_optimization_file(str(path))

    assert exc_info.value.status_code == 422


def test_decode_unsupported_extension():
    with pytest.raises(ValueError, match="unsupported"):
        decode_file_bytes("txt", b"{}")


def test_pickle_forbidden_without_extension_does_not_fall_through(tmp_path):
    path = tmp_path / "noext"
    path.write_bytes(pickle.dumps({"obj": object()}))

    with pytest.raises(HTTPException) as exc_info:
        load_optimization_file(str(path))

    assert exc_info.value.status_code == 422
    assert "forbidden" in exc_info.value.detail
