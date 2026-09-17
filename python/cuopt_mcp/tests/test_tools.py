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
    def __init__(self, solution=None, not_ready_logs=False):
        self.solution = solution
        self.cancelled = []
        self.deleted = []
        self.submitted = []
        self.not_ready_logs = not_ready_logs

    def submit(self, problem, settings, enable_incumbents=None):
        self.submitted.append(enable_incumbents)
        return "job-new"

    def result(self, job_id, variable_names=None):
        return self.solution

    def cancel(self, job_id):
        self.cancelled.append(job_id)

    def delete(self, job_id):
        self.deleted.append(job_id)

    def incumbents(self, job_id, from_index=0):
        # Real shape: list of {"index", "objective", "assignment"} dicts,
        # not (objective, assignment) tuples -- see tools.incumbents.
        entries = [
            {"index": 0, "objective": 10.0, "assignment": []},
            {"index": 1, "objective": 8.0, "assignment": []},
        ]
        return [e for e in entries if e["index"] >= from_index]

    def logs(self, job_id, from_byte=0):
        if self.not_ready_logs:
            # tools.logs matches by class name, not isinstance, precisely
            # so it needn't import the real cuopt.grpc.linear_programming
            # exception type -- this stand-in exercises that.
            class JobNotReadyError(Exception):
                pass

            raise JobNotReadyError(f"job {job_id} is not complete (QUEUED)")
        # Real shape: list[str], not a joined string -- see tools.logs.
        return [f"line {i}" for i in range(10)]


@pytest.fixture
def fake(monkeypatch):
    def _install(solution=None, not_ready_logs=False):
        stub = FakeClient(solution, not_ready_logs=not_ready_logs)
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


def test_result_empty_variables_list_returns_no_variables(fake):
    """variables=[] must mean "return none", distinct from the omitted
    default -- `if variables:` treated both the same and fell through to
    the full/truncated shaping path instead.
    """
    fake(FakeSolution([1.0, 2.0], names=["x", "y"]))
    out = tools.result("job-1", variables=[])
    assert out["variables"] == {}
    assert "missing_variables" not in out


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


def test_result_rejects_solution_dir_that_is_a_file(
    fake, tmp_path, monkeypatch
):
    """A pre-existing non-directory at CUOPT_MCP_SOLUTION_DIR must be
    rejected up front -- previously it passed the ownership/mode check
    (nothing there tested S_ISDIR) and only failed later as an unguarded
    NotADirectoryError out of _write_solution_file.
    """
    stray_file = tmp_path / "not-a-dir"
    stray_file.write_text("")
    stray_file.chmod(0o600)
    monkeypatch.setenv("CUOPT_MCP_SOLUTION_DIR", str(stray_file))
    fake(
        FakeSolution(
            [float(i) for i in range(5000)],
            names=[f"x{i}" for i in range(5000)],
        )
    )
    with pytest.raises(client.CuOptMCPError, match="not a private directory"):
        tools.result("job-big", limit=10)


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
    assert out["ready"] is True
    assert out["lines"] == ["line 7", "line 8", "line 9"]
    assert out["truncated"] is True
    assert out["next_byte"] > 0


def test_logs_reports_not_ready_without_raising(fake):
    fake(not_ready_logs=True)
    out = tools.logs("job-1")
    assert out["ready"] is False
    assert "cuopt_status" in out["hint"]


def test_cancel(fake):
    stub = fake()
    assert tools.cancel("job-1")["cancelled"] is True
    assert stub.cancelled == ["job-1"]


def test_delete(fake):
    stub = fake()
    assert tools.delete("job-1")["deleted"] is True
    assert stub.deleted == ["job-1"]


class FakeModel:
    def get_variable_lower_bounds(self):
        return [0.0, 0.0]

    def get_constraint_matrix_offsets(self):
        return [0, 1]


def test_submit_enables_incumbents_for_mip_only(fake, monkeypatch, tmp_path):
    """Client.submit() only enables incumbent collection by default when
    settings already carries a local MIP callback -- this process keeps
    none, so tools.submit must pass enable_incumbents explicitly, or
    cuopt_incumbents always comes back empty for a MIP job (tools.py).
    """
    stub = fake()
    monkeypatch.setattr(tools, "_read_problem", lambda path: FakeModel())
    monkeypatch.setattr(
        tools, "_build_settings", lambda kind, settings: object()
    )
    problem = tmp_path / "p.mps"
    problem.write_text("")
    tools.submit(str(problem), "mip_settings")
    tools.submit(str(problem), "pdlp_settings")
    assert stub.submitted == [True, False]


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


@pytest.mark.parametrize("bad_port", ["0", "-1", "65536", "999999"])
def test_endpoint_rejects_out_of_range_port(monkeypatch, bad_port):
    monkeypatch.setenv("CUOPT_REMOTE_PORT", bad_port)
    with pytest.raises(client.CuOptMCPError, match="between 1 and 65535"):
        client.endpoint()
