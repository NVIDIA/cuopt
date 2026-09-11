# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool behaviour with a stubbed gRPC client.

These run without a GPU or a cuopt_grpc_server; the live path is covered by
test_end_to_end.py.
"""

import pytest

from cuopt_mcp import client, tools


class FakeSolution:
    def __init__(self, values, names=None):
        self._values = values
        self._names = names

    def get_primal_solution(self):
        return self._values

    def get_vars(self):
        return dict(zip(self._names, self._values)) if self._names else {}

    def get_primal_objective(self):
        return 42.0

    def get_solve_time(self):
        return 1.5

    def get_termination_status(self):
        import enum

        class LPTerminationStatus(enum.IntEnum):
            Optimal = 1

        return LPTerminationStatus.Optimal

    def get_termination_reason(self):
        return "Optimal solution found"


class FakeClient:
    def __init__(self, solution=None):
        self.solution = solution
        self.cancelled = []

    def result(self, job_id, variable_names=None):
        return self.solution

    def cancel(self, job_id):
        self.cancelled.append(job_id)

    def incumbents(self, job_id, from_index=0):
        return [(10.0, None), (8.0, None)][from_index:]

    def logs(self, job_id, from_byte=0):
        return "\n".join(f"line {i}" for i in range(10))


@pytest.fixture
def fake(monkeypatch):
    def _install(solution=None):
        stub = FakeClient(solution)
        monkeypatch.setattr(tools, "get_client", lambda: stub)
        return stub

    yield _install
    client.reset_client()


def test_result_reports_not_ready_without_raising(fake):
    fake(None)
    out = tools.result("job-1")
    assert out["ready"] is False
    assert "cuopt_status" in out["hint"]


def test_result_returns_summary_and_named_values(fake):
    fake(FakeSolution([1.0, 0.0, 3.0], names=["x", "y", "z"]))
    out = tools.result("job-1")
    assert out["primal_objective"] == 42.0
    # IntEnum: str() would give "1", which tells a caller nothing.
    assert out["termination_status"] == "Optimal"
    assert out["termination_status_code"] == 1
    assert out["variables"] == {"x": 1.0, "y": 0.0, "z": 3.0}


def test_result_nonzero_only_filters(fake):
    fake(FakeSolution([1.0, 0.0, 3.0], names=["x", "y", "z"]))
    out = tools.result("job-1", nonzero_only=True)
    assert out["variables"] == {"x": 1.0, "z": 3.0}
    assert out["num_nonzero"] == 2


def test_result_named_lookup_reports_missing(fake):
    fake(FakeSolution([1.0, 2.0], names=["x", "y"]))
    out = tools.result("job-1", variables=["x", "nope"])
    assert out["variables"] == {"x": 1.0}
    assert out["missing_variables"] == ["nope"]


def test_large_solution_is_written_to_file_not_inlined(
    fake, tmp_path, monkeypatch
):
    """A big solution must not be returned inline.

    The binding limit is the model's context window, so past `limit` the
    values go to a file and only a pointer comes back.
    """
    monkeypatch.setenv("CUOPT_MCP_SOLUTION_DIR", str(tmp_path))
    n = 5000
    fake(
        FakeSolution(
            [float(i) for i in range(n)], names=[f"x{i}" for i in range(n)]
        )
    )
    out = tools.result("job-big", limit=10)
    assert out["variables_truncated"] is True
    assert len(out["variables"]) == 10
    assert out["num_variables"] == n
    written = tmp_path / "job-big.json"
    assert written.is_file()
    import json

    assert len(json.loads(written.read_text())) == n


def test_unnamed_solution_falls_back_to_indices_with_a_hint(fake):
    fake(FakeSolution([1.0, 2.0]))
    out = tools.result("job-1")
    assert out["variables"] == {"0": 1.0, "1": 2.0}
    assert "names_from" in out["names"]


def test_incumbents_paginate(fake):
    fake()
    out = tools.incumbents("job-1", from_index=1)
    assert out["count"] == 1
    assert out["incumbents"][0]["index"] == 1
    assert out["next_index"] == 2


def test_logs_tail_is_bounded(fake):
    fake()
    out = tools.logs("job-1", tail_lines=3)
    assert out["lines"] == ["line 7", "line 8", "line 9"]
    assert out["truncated"] is True


def test_cancel(fake):
    stub = fake()
    assert tools.cancel("job-1")["cancelled"] is True
    assert stub.cancelled == ["job-1"]


def test_missing_problem_file_is_a_clear_error():
    with pytest.raises(client.CuOptMCPError, match="problem file not found"):
        tools.submit("/nonexistent/model.mps", "pdlp_settings")


def test_list_settings_names_and_detail():
    listing = tools.list_settings("pdlp_settings")
    assert "time_limit" in listing["parameters"]
    detail = tools.list_settings("pdlp_settings", name="pdlp_solver_mode")
    assert "Stable3" in detail["enum"]


def test_list_settings_rejects_bad_kind():
    with pytest.raises(client.CuOptMCPError, match="mip_settings"):
        tools.list_settings("nonsense")


def test_unreachable_server_message_names_the_endpoint(monkeypatch):
    monkeypatch.setenv("CUOPT_REMOTE_HOST", "gpu-host")
    monkeypatch.setenv("CUOPT_REMOTE_PORT", "50999")
    err = client.describe_connection_error(RuntimeError("UNAVAILABLE"))
    assert "gpu-host:50999" in str(err)
