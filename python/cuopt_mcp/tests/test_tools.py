# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool behaviour with a stubbed gRPC client.

These run without a GPU or a cuopt_grpc_server; the live path is covered by
test_end_to_end.py.
"""

import pytest

from cuopt_mcp import client, tools
from cuopt_mcp.client import CuOptMCPError


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
    def __init__(self, solution=None, status_error=None):
        self.solution = solution
        self.cancelled = []
        self.status_error = status_error
        self.probed = []

    def status(self, job_id):
        self.probed.append(job_id)
        if self.status_error is not None:
            raise self.status_error
        import enum

        class JobStatus(enum.IntEnum):
            NOT_FOUND = 5

        return JobStatus.NOT_FOUND

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
    def _install(solution=None, status_error=None):
        stub = FakeClient(solution, status_error)
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


def test_unreachable_message_says_to_look_before_starting_a_server():
    """The advice must not read as 'start one', full stop.

    Told only to start a server, a caller that already has one running
    elsewhere starts a second. Two servers can share a listen port, after
    which jobs and result lookups land in different processes.
    """
    err = str(client.describe_connection_error(RuntimeError("UNAVAILABLE")))
    assert "pgrep" in err
    assert err.index("already running") < err.index("start one with")


def test_health_names_the_endpoint_and_probes_it(fake, monkeypatch):
    monkeypatch.setenv("CUOPT_REMOTE_HOST", "gpu-host")
    monkeypatch.setenv("CUOPT_REMOTE_PORT", "50999")
    stub = fake()
    out = tools.health()
    assert (out["host"], out["port"]) == ("gpu-host", 50999)
    assert out["reachable"] is True
    # A NOT_FOUND answer still proves a server answered.
    assert stub.probed == [tools.PROBE_JOB_ID]


def test_health_reports_unreachable_without_raising(fake, monkeypatch):
    monkeypatch.setenv("CUOPT_REMOTE_HOST", "gpu-host")
    monkeypatch.setenv("CUOPT_REMOTE_PORT", "50999")
    fake(status_error=RuntimeError("failed to connect to all addresses"))
    out = tools.health()
    assert out["reachable"] is False
    assert "gpu-host:50999" in out["error"]
    assert "pgrep" in out["error"]


def test_health_drops_a_dead_channel(fake, monkeypatch):
    """A cached channel outlives the failure, so every later call would fail."""
    dropped = []
    monkeypatch.setattr(tools, "reset_client", lambda: dropped.append(True))
    fake(status_error=RuntimeError("UNAVAILABLE"))
    tools.health()
    assert dropped == [True]


# --- JSON model entry point -------------------------------------------


def _base_problem():
    return {
        "objective": [1.0, 1.0],
        "constraint_matrix": {
            "rows": [0, 0, 1],
            "cols": [0, 1, 0],
            "values": [1.0, 1.0, 1.0],
        },
        "constraint_upper_bounds": [4.0, 3.0],
    }


def test_json_coo_is_converted_to_csr():
    model = tools._build_model_from_json(_base_problem())
    assert list(model.get_constraint_matrix_offsets()) == [0, 2, 3]
    assert list(model.get_constraint_matrix_indices()) == [0, 1, 0]
    assert len(model.get_variable_lower_bounds()) == 2


def test_json_accepts_csr_directly():
    problem = _base_problem()
    problem["constraint_matrix"] = {
        "offsets": [0, 2, 3],
        "indices": [0, 1, 0],
        "values": [1.0, 1.0, 1.0],
    }
    model = tools._build_model_from_json(problem)
    assert list(model.get_constraint_matrix_offsets()) == [0, 2, 3]


def test_null_bound_becomes_infinite():
    problem = _base_problem()
    problem["constraint_lower_bounds"] = [None, 2.0]
    model = tools._build_model_from_json(problem)
    lower = model.get_constraint_lower_bounds()
    assert lower[0] == float("-inf") and lower[1] == 2.0


def test_large_sentinel_is_normalised_to_infinity():
    # A finite 1e30 row bound can make cuOpt return a constraint-violating
    # point reported as Optimal, so it must not survive as a finite value.
    problem = _base_problem()
    problem["constraint_lower_bounds"] = [-1e30, -1e30]
    lower = tools._build_model_from_json(problem).get_constraint_lower_bounds()
    assert all(v == float("-inf") for v in lower)


def test_integrality_does_not_disturb_bounds():
    # The MPS INTORG/INTEND trap: an integer column with no explicit bound
    # silently becomes [0, 1]. variable_types must not do that.
    problem = _base_problem()
    problem["variable_types"] = ["I", "I"]
    model = tools._build_model_from_json(problem)
    assert list(model.get_variable_types()) == ["I", "I"]
    assert all(v == float("inf") for v in model.get_variable_upper_bounds())


def test_variable_names_round_trip():
    problem = _base_problem()
    problem["variable_names"] = ["x", "y"]
    assert list(
        tools._build_model_from_json(problem).get_variable_names()
    ) == [
        "x",
        "y",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"problem_path": "a.mps", "problem": {"objective": [1.0]}},
    ],
)
def test_submit_requires_exactly_one_model_source(kwargs):
    with pytest.raises(CuOptMCPError, match="exactly one"):
        tools.submit(kind="mip_settings", **kwargs)


@pytest.mark.parametrize(
    "problem,message",
    [
        (
            {"constraint_matrix": {"rows": [], "cols": [], "values": []}},
            "missing required key 'objective'",
        ),
        ({"objective": [1.0]}, "missing required key 'constraint_matrix'"),
        (
            {
                "objective": [1.0, 1.0],
                "constraint_matrix": {
                    "rows": [0],
                    "cols": [0],
                    "values": [1.0, 2.0],
                },
            },
            "equal length",
        ),
        (
            {
                "objective": [1.0],
                "constraint_matrix": {
                    "rows": [0],
                    "cols": [5],
                    "values": [1.0],
                },
            },
            "only 1 variables",
        ),
    ],
)
def test_json_model_errors_name_the_offending_key(problem, message):
    with pytest.raises(CuOptMCPError, match=message):
        tools._build_model_from_json(problem)


def test_variable_types_rejects_unknown_code():
    problem = _base_problem()
    problem["variable_types"] = ["I", "B"]
    with pytest.raises(CuOptMCPError, match="'C' or 'I'"):
        tools._build_model_from_json(problem)


def test_nonzero_only_drops_numerical_dust():
    dust = {"a": 1.0, "b": 3.4e-13, "c": -1.5e-12}
    assert {k: v for k, v in dust.items() if abs(v) > tools.ZERO_TOL} == {
        "a": 1.0
    }


def test_duplicate_cells_are_summed():
    # Building a row incrementally emits a cell twice; the row must mean
    # 2x <= 10, not depend on how the backend treats repeated indices.
    model = tools._build_model_from_json(
        {
            "objective": [1.0],
            "constraint_matrix": {
                "rows": [0, 0],
                "cols": [0, 0],
                "values": [1.0, 1.0],
            },
            "constraint_upper_bounds": [10.0],
        }
    )
    assert list(model.get_constraint_matrix_offsets()) == [0, 1]
    assert list(model.get_constraint_matrix_indices()) == [0]
    assert list(model.get_constraint_matrix_values()) == [2.0]


def test_duplicate_cells_summed_across_interleaved_rows():
    model = tools._build_model_from_json(
        {
            "objective": [1.0, 1.0],
            "constraint_matrix": {
                "rows": [1, 0, 1, 0, 1],
                "cols": [0, 1, 0, 1, 1],
                "values": [3.0, 1.0, 4.0, 2.0, 5.0],
            },
            "constraint_upper_bounds": [1.0, 1.0],
        }
    )
    assert list(model.get_constraint_matrix_offsets()) == [0, 1, 3]
    assert list(model.get_constraint_matrix_indices()) == [1, 0, 1]
    assert list(model.get_constraint_matrix_values()) == [3.0, 7.0, 5.0]


def test_trailing_all_zero_row_is_preserved():
    # Row 1 has no entries. Inferring the count from the data would drop it
    # and then blame the bounds array for the length mismatch.
    model = tools._build_model_from_json(
        {
            "objective": [1.0, 1.0],
            "constraint_matrix": {
                "rows": [0],
                "cols": [0],
                "values": [1.0],
            },
            "constraint_upper_bounds": [5.0, 7.0],
            "constraint_lower_bounds": [None, None],
        }
    )
    assert list(model.get_constraint_matrix_offsets()) == [0, 1, 1]
    assert len(model.get_constraint_upper_bounds()) == 2


def test_row_index_beyond_declared_count_blames_the_matrix():
    with pytest.raises(CuOptMCPError, match="references row 2 but only 2"):
        tools._build_model_from_json(
            {
                "objective": [1.0],
                "constraint_matrix": {
                    "rows": [0, 2],
                    "cols": [0, 0],
                    "values": [1.0, 1.0],
                },
                "constraint_upper_bounds": [5.0, 7.0],
            }
        )


def test_mismatched_constraint_bound_lengths_are_reported():
    with pytest.raises(CuOptMCPError, match="same length"):
        tools._build_model_from_json(
            {
                "objective": [1.0],
                "constraint_matrix": {
                    "rows": [0],
                    "cols": [0],
                    "values": [1.0],
                },
                "constraint_lower_bounds": [0.0],
                "constraint_upper_bounds": [5.0, 7.0],
            }
        )


def test_explicit_n_constraints_allows_a_fully_empty_row_block():
    model = tools._build_model_from_json(
        {
            "objective": [1.0],
            "constraint_matrix": {"rows": [], "cols": [], "values": []},
            "n_constraints": 3,
        }
    )
    assert list(model.get_constraint_matrix_offsets()) == [0, 0, 0, 0]


def test_csr_row_count_must_match_declared_bounds():
    with pytest.raises(CuOptMCPError, match="has 1 rows but 2 constraint"):
        tools._build_model_from_json(
            {
                "objective": [1.0],
                "constraint_matrix": {
                    "offsets": [0, 1],
                    "indices": [0],
                    "values": [1.0],
                },
                "constraint_upper_bounds": [5.0, 7.0],
            }
        )
