# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""routing.py: JSON -> DataModel mapping and tool behavior with a stubbed
gRPC client.

These run without a GPU or a cuopt_grpc_server; the live path is covered by
test_end_to_end.py. build_model tests need cuopt.routing importable
(pytestmark below), unlike the stubbed-client tests, which stub the client
entirely and never touch cuopt.routing.
"""

import pytest

from cuopt_mcp import client, routing

cuopt_routing = pytest.importorskip(
    "cuopt.routing",
    reason="cuopt.routing (and its cuDF/CUDA chain) not importable",
)


UUID1 = "11111111-1111-1111-1111-111111111111"

MIN_PROBLEM = {
    "n_locations": 2,
    "fleet_size": 1,
    "cost_matrices": [{"values": [[0, 1], [1, 0]]}],
}


def test_build_model_requires_n_locations():
    with pytest.raises(client.CuOptMCPError, match="n_locations"):
        routing._build_routing_model_from_json({"fleet_size": 1})


def test_build_model_requires_cost_matrices():
    with pytest.raises(client.CuOptMCPError, match="cost_matrices"):
        routing._build_routing_model_from_json(
            {"n_locations": 2, "fleet_size": 1}
        )


def test_build_model_rejects_empty_cost_matrices():
    with pytest.raises(client.CuOptMCPError, match="cost_matrices"):
        routing._build_routing_model_from_json(
            {"n_locations": 2, "fleet_size": 1, "cost_matrices": []}
        )


def test_build_model_minimal():
    dm = routing._build_routing_model_from_json(MIN_PROBLEM)
    assert dm.get_num_locations() == 2
    assert dm.get_fleet_size() == 1
    assert dm.get_num_orders() == 2  # defaults to n_locations


def test_build_model_rejects_mutually_exclusive_breaks():
    problem = {
        **MIN_PROBLEM,
        "uniform_breaks": [{"earliest": [0], "latest": [10], "duration": [1]}],
        "vehicle_breaks": [
            {"vehicle_id": 0, "earliest": 0, "latest": 10, "duration": 1}
        ],
    }
    with pytest.raises(client.CuOptMCPError, match="mutually exclusive"):
        routing._build_routing_model_from_json(problem)


def test_build_model_rejects_bad_objective_name():
    problem = {
        **MIN_PROBLEM,
        "objective": {
            "objectives": ["NOT_A_REAL_OBJECTIVE"],
            "weights": [1.0],
        },
    }
    with pytest.raises(client.CuOptMCPError, match="NOT_A_REAL_OBJECTIVE"):
        routing._build_routing_model_from_json(problem)


def test_build_model_rejects_bad_node_type_name():
    problem = {
        **MIN_PROBLEM,
        "initial_solutions": {
            "vehicle_ids": [0],
            "routes": [1],
            "types": ["NotAType"],
            "sol_offsets": [0],
        },
    }
    with pytest.raises(client.CuOptMCPError, match="NotAType"):
        routing._build_routing_model_from_json(problem)


def test_build_model_full_feature_set():
    """Exercises every JSON key routing.py maps, verified via
    problem_summary (the same _populate path submit() uses) rather than a
    live server -- catches a wrong setter/arg mapping without a GPU.
    """
    from cuopt.grpc.routing.grpc_client import problem_summary

    problem = {
        "n_locations": 5,
        "fleet_size": 2,
        "n_orders": 3,
        "cost_matrices": [
            {"values": [[0, 1, 2, 3, 4]] * 5, "vehicle_type": 0},
            {"values": [[0, 2, 4, 6, 8]] * 5, "vehicle_type": 1},
        ],
        "transit_time_matrices": [{"values": [[0, 1, 2, 3, 4]] * 5}],
        "vehicle_types": [0, 1],
        "vehicle_locations": {"start": [0, 0], "end": [0, 0]},
        "vehicle_time_windows": {"earliest": [0, 0], "latest": [100, 100]},
        "drop_return_trips": [False, True],
        "skip_first_trips": [False, False],
        "vehicle_max_costs": [50.0, 50.0],
        "vehicle_max_times": [80.0, 80.0],
        "vehicle_fixed_costs": [1.0, 2.0],
        "order_locations": [1, 2, 3],
        "order_time_windows": {"earliest": [0, 0, 0], "latest": [90, 90, 90]},
        "order_prizes": [1.0, 2.0, 3.0],
        "order_service_times": [
            {"service_times": [1, 1, 1]},
            {"service_times": [2, 2, 2], "vehicle_id": 0},
        ],
        "pickup_delivery_pairs": {"pickup": [0], "delivery": [1]},
        "capacity_dimensions": [
            {"name": "weight", "demand": [1, 1, 1], "capacity": [5, 5]}
        ],
        "break_locations": [0],
        "vehicle_breaks": [
            {"vehicle_id": 0, "earliest": 10, "latest": 20, "duration": 5}
        ],
        "vehicle_distance_breaks": [
            {
                "vehicle_id": 1,
                "distance_min": 1.0,
                "distance_max": 5.0,
                "duration": 5,
            }
        ],
        "vehicle_order_match": [{"vehicle_id": 0, "orders": [0, 1]}],
        "order_vehicle_match": [{"order_id": 2, "vehicles": [0, 1]}],
        "order_precedence": [{"order_id": 2, "preceding_orders": [0]}],
        "objective": {
            "objectives": ["cost", "travel_time"],
            "weights": [1.0, 0.5],
        },
        "min_vehicles": 1,
        "initial_solutions": {
            "vehicle_ids": [0],
            "routes": [1],
            "types": ["delivery"],
            "sol_offsets": [0],
        },
    }
    dm = routing._build_routing_model_from_json(problem)
    summary = problem_summary(dm)
    assert summary["num_locations"] == 5
    assert summary["fleet_size"] == 2
    assert summary["num_orders"] == 3
    assert summary["cost_matrices"] == 2
    assert summary["transit_time_matrices"] == 1
    assert summary["vehicle_start_locations"] == 2
    assert summary["vehicle_tw_earliest"] == 2
    assert summary["order_locations"] == 3
    assert summary["order_tw_earliest"] == 3
    assert summary["order_prizes"] == 3
    assert summary["order_service_times"] == 2
    assert summary["pickup_indices"] == 1
    assert summary["capacity_dimensions"] == 1
    assert summary["break_locations"] == 1
    assert summary["vehicle_breaks"] == 1
    assert summary["vehicle_distance_breaks"] == 1
    assert summary["vehicle_order_match"] == 1
    assert summary["order_vehicle_match"] == 1
    assert summary["order_precedence"] == 1
    assert summary["objectives"] == 2
    assert summary["min_vehicles"] == 1
    assert summary["initial_solutions_routes"] == 1


class FakeRoutingClient:
    def __init__(self, solution=None):
        self.solution = solution
        self.submitted = []
        self.deleted = []

    def submit(self, data_model, settings=None):
        self.submitted.append((data_model, settings))
        return "job-new"

    def result(self, job_id):
        return self.solution

    def delete(self, job_id):
        self.deleted.append(job_id)


@pytest.fixture
def fake_routing(monkeypatch):
    def _install(solution=None):
        stub = FakeRoutingClient(solution)
        monkeypatch.setattr(routing, "get_routing_client", lambda: stub)
        return stub

    yield _install
    client.reset_routing_client()


def test_submit_passes_settings_through(fake_routing):
    stub = fake_routing()
    out = routing.submit(MIN_PROBLEM, settings={"time_limit": 5.0})
    assert out["job_id"] == "job-new"
    assert out["num_locations"] == 2
    assert out["fleet_size"] == 1
    assert stub.submitted[0][1] == {"time_limit": 5.0}


def test_result_reports_not_ready_without_raising(fake_routing):
    fake_routing(None)
    out = routing.result("job-1")
    assert out["ready"] is False
    assert "cuopt_status" in out["hint"]


def _fake_solution(**overrides):
    sol = {
        "status": 0,
        "status_message": "cuOpt solver success.",
        "error_message": "",
        "vehicle_count": 1,
        "total_objective_value": 7.0,
        "objective_values": {0: 7.0},
        "truck_id": [0, 0, 0],
        "locations": [0, 1, 0],
        "node_types": [0, 2, 0],
        "arrival_stamp": [0.0, 1.0, 2.0],
        "unserviced_nodes": [],
    }
    sol.update(overrides)
    return sol


def test_result_shapes_a_successful_solution(fake_routing):
    fake_routing(_fake_solution())
    out = routing.result("job-1")
    assert out["ready"] is True
    assert out["status"] == "SUCCESS"
    assert out["objective_values"] == {"COST": 7.0}
    assert out["stops"] == [
        {"vehicle": 0, "location": 0, "type": "Depot", "arrival": 0.0},
        {"vehicle": 0, "location": 1, "type": "Delivery", "arrival": 1.0},
        {"vehicle": 0, "location": 0, "type": "Depot", "arrival": 2.0},
    ]
    assert "unserviced_orders" not in out


def test_result_reports_unserviced_orders_on_fail(fake_routing):
    fake_routing(
        _fake_solution(
            status=1,
            status_message="",
            error_message="infeasible",
            unserviced_nodes=[1, 2],
        )
    )
    out = routing.result("job-1")
    assert out["status"] == "FAIL"
    assert out["error_message"] == "infeasible"
    assert out["unserviced_orders"] == [1, 2]


def test_result_truncates_large_solution_to_a_file(
    fake_routing, tmp_path, monkeypatch
):
    monkeypatch.setenv("CUOPT_MCP_SOLUTION_DIR", str(tmp_path))
    n = 20
    fake_routing(
        _fake_solution(
            truck_id=[0] * n,
            locations=list(range(n)),
            node_types=[0] * n,
            arrival_stamp=[float(i) for i in range(n)],
        )
    )
    out = routing.result(UUID1, limit=5)
    assert out["stops_truncated"] is True
    assert len(out["stops"]) == 5
    written = tmp_path / f"{UUID1}.vrp.json"
    assert written.is_file()
    import json

    assert len(json.loads(written.read_text())) == n


def test_result_rejects_bad_limit(fake_routing):
    fake_routing(_fake_solution())
    with pytest.raises(client.CuOptMCPError, match="limit"):
        routing.result("job-1", limit=-1)
