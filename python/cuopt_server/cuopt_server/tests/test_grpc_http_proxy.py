# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cuopt_server.cuopt_proxy import parse_args
from cuopt_server.proxy_webserver import (
    app,
    reset_proxy_state,
    set_grpc_client,
    set_max_request_size,
)
from cuopt_server.utils.http_codec import mime_json, mime_msgpack, mime_zlib
from cuopt_server.utils.http_envelope import make_response
from cuopt_server.utils.linear_programming import conversion as lp_conversion


def _lp():
    return {
        "csr_constraint_matrix": {
            "offsets": [0, 2],
            "indices": [0, 1],
            "values": [1.0, 1.0],
        },
        "constraint_bounds": {
            "upper_bounds": [5000.0],
            "lower_bounds": [0.0],
        },
        "objective_data": {
            "coefficients": [1.2, 1.7],
            "scalability_factor": 1.0,
            "offset": 0.0,
        },
        "variable_bounds": {
            "upper_bounds": [3000.0, 5000.0],
            "lower_bounds": [0.0, 0.0],
        },
        "maximize": False,
        "variable_names": ["x", "y"],
        "solver_config": {"time_limit": 5},
    }


class FakeJobStatus:
    QUEUED = SimpleNamespace(name="QUEUED")
    PROCESSING = SimpleNamespace(name="PROCESSING")
    COMPLETED = SimpleNamespace(name="COMPLETED")
    FAILED = SimpleNamespace(name="FAILED")
    CANCELLED = SimpleNamespace(name="CANCELLED")
    NOT_FOUND = SimpleNamespace(name="NOT_FOUND")


class FakeSol:
    def get_termination_status(self):
        from cuopt.linear_programming.solver.solver_wrapper import (
            LPTerminationStatus,
        )

        return LPTerminationStatus.Optimal

    def get_primal_solution(self):
        import numpy as np

        return np.array([0.0, 0.0])

    def get_dual_solution(self):
        import numpy as np

        return np.array([0.0])

    def get_lp_stats(self):
        return {"gap": 0.0}

    def get_reduced_cost(self):
        import numpy as np

        return np.array([1.2, 1.7])

    def get_milp_stats(self):
        raise AttributeError

    def get_pdlp_warm_start_data(self):
        raise AttributeError

    def get_problem_category(self):
        return SimpleNamespace(name="LP")

    def get_primal_objective(self):
        return 0.0

    def get_dual_objective(self):
        return 0.0

    def get_solve_time(self):
        return 0.01

    def get_solved_by(self):
        return SimpleNamespace(name="PDLP")

    def get_vars(self):
        return {"x": 0.0, "y": 0.0}

    def get_termination_reason(self):
        return "Optimal"


class FakeClient:
    def __init__(self):
        self.jobs = {}
        self.submitted = []
        self.cancelled = []
        self.deleted = []
        self._incumbents = {}
        self._logs = {}

    def submit(self, problem, settings, enable_incumbents=None):
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = FakeJobStatus.COMPLETED
        self.submitted.append(
            {
                "id": job_id,
                "enable_incumbents": enable_incumbents,
                "problem": problem,
                "settings": settings,
            }
        )
        return job_id

    def status(self, job_id):
        return self.jobs.get(job_id, FakeJobStatus.NOT_FOUND)

    def result(self, job_id, variable_names=None):
        status = self.jobs.get(job_id)
        if status in (FakeJobStatus.FAILED, FakeJobStatus.CANCELLED):
            raise RuntimeError(f"job {status.name.lower()}")
        if status != FakeJobStatus.COMPLETED:
            return None
        return FakeSol()

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        if job_id in self.jobs:
            self.jobs[job_id] = FakeJobStatus.CANCELLED

    def delete(self, job_id):
        self.deleted.append(job_id)
        self.jobs.pop(job_id, None)

    def logs(self, job_id, from_byte=0):
        return list(self._logs.get(job_id, ["line1", "line2"]))

    def incumbents(self, job_id, from_index=0):
        entries = self._incumbents.get(job_id, [])
        return [e for e in entries if e["index"] >= from_index]


@pytest.fixture
def proxy(monkeypatch):
    import cuopt_server.proxy_webserver as pw

    reset_proxy_state()
    set_max_request_size(1024 * 1024 * 1024)
    fake = FakeClient()
    set_grpc_client(fake)
    monkeypatch.setattr(
        pw, "create_data_model", lambda lp: ([], SimpleNamespace())
    )
    monkeypatch.setattr(
        pw, "create_solver", lambda lp, w: ([], SimpleNamespace())
    )
    client = TestClient(app)
    yield client, fake
    reset_proxy_state()
    set_max_request_size(1024 * 1024 * 1024)


def test_parse_args_defaults():
    args = parse_args([])
    assert args.port == 8000
    assert args.grpc_host == "127.0.0.1"
    assert args.grpc_port == 5001


def test_parse_args_overrides():
    args = parse_args(
        [
            "--ip",
            "127.0.0.1",
            "-p",
            "9000",
            "--grpc-host",
            "gpu",
            "--grpc-port",
            "5002",
        ]
    )
    assert args.ip == "127.0.0.1"
    assert args.port == 9000
    assert args.grpc_host == "gpu"
    assert args.grpc_port == 5002


@pytest.mark.parametrize(
    "variable, option",
    [
        ("CUOPT_SERVER_PORT", "--port"),
        ("CUOPT_GRPC_PORT", "--grpc-port"),
        ("CUOPT_MAX_RESULT", "--max-result"),
        ("CUOPT_MAX_REQUEST_SIZE", "--max-request-size"),
    ],
)
def test_parse_args_rejects_invalid_numeric_environment(
    monkeypatch, variable, option, capsys
):
    monkeypatch.setenv(variable, "invalid")
    with pytest.raises(SystemExit):
        parse_args([])
    assert option in capsys.readouterr().err


def test_make_response_envelope():
    r = make_response(
        {"solver_response": {"status": "Optimal"}},
        warnings=["w"],
        notes=["n"],
        reqId="abc",
        total_solve_time=1.5,
    )
    assert r["reqId"] == "abc"
    assert r["warnings"] == ["w"]
    assert r["notes"] == ["n"]
    assert r["response"]["total_solve_time"] == 1.5


def test_solution_to_legacy_http_strips_warmstart():
    res = lp_conversion.solution_to_legacy_http(FakeSol())
    assert res["status"] == "Optimal"
    assert "pdlpwarmstart_data" in res["solution"]
    stripped = lp_conversion.solution_to_legacy_http(
        FakeSol(), include_warmstart=False
    )
    assert "pdlpwarmstart_data" not in stripped["solution"]


def test_health(proxy):
    client, _ = proxy
    for path in ("/", "/cuopt/health", "/v2/health/ready", "/v2/health/live"):
        res = client.get(path)
        assert res.status_code == 200, path
        body = res.json()
        assert body["status"] == "RUNNING"
        assert "version" in body


def test_submit_status_result_delete(proxy):
    client, fake = proxy
    lp = _lp()
    res = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom", "Content-Type": mime_json},
        json=lp,
    )
    assert res.status_code == 200, res.text
    req_id = res.json()["reqId"]
    uuid.UUID(req_id)
    assert fake.submitted[0]["enable_incumbents"] is False

    st = client.get(f"/cuopt/request/{req_id}")
    assert st.status_code == 200
    assert st.json() == "completed"

    sol = client.get(f"/cuopt/solution/{req_id}")
    assert sol.status_code == 200
    body = sol.json()
    assert body["reqId"] == req_id
    assert body["response"]["solver_response"]["status"] == "Optimal"
    assert (
        "pdlpwarmstart_data"
        not in body["response"]["solver_response"]["solution"]
    )

    deleted = client.delete(f"/cuopt/solution/{req_id}")
    assert deleted.status_code == 200
    assert req_id in fake.deleted


def test_delete_preserves_metadata_when_grpc_delete_fails(proxy, monkeypatch):
    import cuopt_server.proxy_webserver as pw

    client, fake = proxy
    req_id = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        json=_lp(),
    ).json()["reqId"]

    def fail_delete(_):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(fake, "delete", fail_delete)
    assert client.delete(f"/cuopt/solution/{req_id}").status_code == 500
    assert pw._get_job(req_id) is not None


@pytest.mark.parametrize(
    "mutate, status_code",
    [
        (lambda lp: lp.pop("csr_constraint_matrix"), 422),
        (
            lambda lp: lp["csr_constraint_matrix"].update(values=[1.0]),
            400,
        ),
        (
            lambda lp: lp["variable_bounds"].update(lower_bounds=[0.0]),
            400,
        ),
    ],
)
def test_invalid_lp_payloads_are_rejected(proxy, mutate, status_code):
    client, fake = proxy
    lp = _lp()
    mutate(lp)
    res = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        json=lp,
    )
    assert res.status_code == status_code, res.text
    assert fake.submitted == []


def test_oversized_request_is_rejected_before_allocation(proxy):
    client, fake = proxy
    set_max_request_size(1)
    res = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        json=_lp(),
    )
    assert res.status_code == 413
    assert fake.submitted == []


def test_incumbents_cursor_and_sentinel(proxy):
    client, fake = proxy
    lp = _lp()
    lp["variable_types"] = ["I", "I"]
    res = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        params={"incumbent_solutions": True},
        json=lp,
    )
    req_id = res.json()["reqId"]
    assert fake.submitted[0]["enable_incumbents"] is True
    fake._incumbents[req_id] = [
        {"index": 0, "objective": 2.0, "assignment": [1.0, 1.0]},
        {"index": 1, "objective": 1.0, "assignment": [0.0, 1.0]},
    ]
    first = client.get(f"/cuopt/solution/{req_id}/incumbents")
    assert first.status_code == 200
    assert first.json() == [
        {"solution": [1.0, 1.0], "cost": 2.0, "bound": None},
        {"solution": [0.0, 1.0], "cost": 1.0, "bound": None},
    ]
    second = client.get(f"/cuopt/solution/{req_id}/incumbents")
    assert second.json() == [{"solution": [], "cost": None, "bound": None}]


def test_logs_and_log_delete_noop(proxy):
    client, fake = proxy
    lp = _lp()
    res = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        params={"solver_logs": True},
        json=lp,
    )
    req_id = res.json()["reqId"]
    logs = client.get(f"/cuopt/log/{req_id}")
    assert logs.status_code == 200
    body = logs.json()
    assert body["log"] == ["line1", "line2"]
    assert body["nbytes"] > 0
    assert client.delete(f"/cuopt/log/{req_id}").status_code == 200


def test_cancel_request(proxy):
    client, fake = proxy
    lp = _lp()
    req_id = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        json=lp,
    ).json()["reqId"]
    fake.jobs[req_id] = FakeJobStatus.PROCESSING
    res = client.delete(f"/cuopt/request/{req_id}")
    assert res.status_code == 200
    assert res.json() == {"queued": 0, "running": 1, "cached": 0}
    assert req_id in fake.cancelled


def test_cancel_completed_is_noop(proxy):
    client, fake = proxy
    lp = _lp()
    req_id = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        json=lp,
    ).json()["reqId"]
    res = client.delete(f"/cuopt/request/{req_id}")
    assert res.status_code == 200
    assert res.json() == {"queued": 0, "running": 0, "cached": 0}
    assert req_id not in fake.cancelled


def test_validation_only_skips_submit(proxy):
    client, fake = proxy
    lp = _lp()
    res = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        params={"validation_only": True},
        json=lp,
    )
    assert res.status_code == 200
    req_id = res.json()["reqId"]
    assert fake.submitted == []
    st = client.get(f"/cuopt/request/{req_id}")
    assert st.json() == "completed"
    sol = client.get(f"/cuopt/solution/{req_id}").json()
    assert sol["notes"] == ["Input is valid"]
    assert sol["response"]["solver_response"]["status"] == 0


@pytest.mark.parametrize(
    "params,feature",
    [
        ({"cache": True}, "cache"),
        ({"reqId": str(uuid.uuid4())}, "reqId"),
        ({"initialId": str(uuid.uuid4())}, "initialId"),
        ({"warmstartId": str(uuid.uuid4())}, "warmstartId"),
        ({"incumbent_set_solutions": True}, "incumbent_set_solutions"),
    ],
)
def test_dropped_query_params_are_501(proxy, params, feature):
    client, _ = proxy
    res = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        params=params,
        json=_lp(),
    )
    assert res.status_code == 501, res.text
    assert feature in res.json()["error"]


def test_batch_lp_is_501(proxy):
    client, _ = proxy
    res = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        json=[_lp(), _lp()],
    )
    assert res.status_code == 501
    assert "Batch LP" in res.json()["error"]


def test_vrp_body_is_501(proxy):
    client, _ = proxy
    res = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        json={
            "cost_matrix_data": {"data": {"1": [[0, 1], [1, 0]]}},
            "task_data": {"task_locations": [1, 1]},
        },
    )
    assert res.status_code == 501
    assert "routing" in res.json()["error"].lower()


def test_post_solution_and_warmstart_and_sync_are_501(proxy):
    client, _ = proxy
    assert client.post("/cuopt/solution", json={}).status_code == 501
    assert (
        client.get(f"/cuopt/solution/{uuid.uuid4()}/warmstart").status_code
        == 501
    )
    assert client.post("/cuopt/cuopt", json={}).status_code == 501
    assert client.delete("/cuopt/request/*").status_code == 501
    assert (
        client.delete(
            f"/cuopt/request/{uuid.uuid4()}", params={"running": True}
        ).status_code
        == 501
    )


def test_msgpack_round_trip_headers(proxy):
    import msgpack

    client, _ = proxy
    payload = msgpack.dumps(_lp())
    res = client.post(
        "/cuopt/request",
        headers={
            "CLIENT-VERSION": "custom",
            "Content-Type": mime_msgpack,
            "Accept": mime_msgpack,
        },
        content=payload,
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith(mime_msgpack)
    body = msgpack.loads(res.content, strict_map_key=False)
    assert "reqId" in body


def test_zlib_accept(proxy):
    import zlib

    client, fake = proxy
    res = client.post(
        "/cuopt/request",
        headers={
            "CLIENT-VERSION": "custom",
            "Accept": mime_zlib,
        },
        json=_lp(),
    )
    assert res.status_code == 200
    body = json.loads(zlib.decompress(res.content))
    assert "reqId" in body
    sol = client.get(
        f"/cuopt/solution/{body['reqId']}",
        headers={"Accept": mime_zlib},
    )
    assert sol.status_code == 200
    decoded = json.loads(zlib.decompress(sol.content))
    assert decoded["response"]["solver_response"]["status"] == "Optimal"


def test_unknown_id_is_404(proxy):
    client, _ = proxy
    missing = str(uuid.uuid4())
    assert client.get(f"/cuopt/request/{missing}").status_code == 404
    assert client.get(f"/cuopt/solution/{missing}").status_code == 404


def test_invalid_id_is_400(proxy):
    client, _ = proxy
    assert client.get("/cuopt/request/not-a-uuid").status_code == 400


@pytest.mark.parametrize("status", ["FAILED", "CANCELLED"])
def test_failed_or_cancelled_solution_is_409(proxy, status):
    client, fake = proxy
    req_id = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        json=_lp(),
    ).json()["reqId"]
    fake.jobs[req_id] = getattr(FakeJobStatus, status)
    res = client.get(f"/cuopt/solution/{req_id}")
    assert res.status_code == 409, res.text
    assert status.lower() in res.json()["error"]


def test_lp_does_not_enable_incumbents(proxy):
    client, fake = proxy
    res = client.post(
        "/cuopt/request",
        headers={"CLIENT-VERSION": "custom"},
        params={"incumbent_solutions": True},
        json=_lp(),
    )
    assert res.status_code == 200
    assert fake.submitted[0]["enable_incumbents"] is False


def test_log_delete_error_is_encoded(proxy, monkeypatch):
    import cuopt_server.proxy_webserver as pw

    client, _ = proxy

    def boom(id):
        raise RuntimeError("log delete failed")

    monkeypatch.setattr(pw, "_require_uuid", boom)
    res = client.delete(
        f"/cuopt/log/{uuid.uuid4()}",
        headers={"Accept": mime_msgpack},
    )
    assert res.status_code == 500
    assert res.headers["content-type"].startswith(mime_msgpack)
