# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_server.utils.linear_programming import conversion
from cuopt_server.utils.linear_programming.data_definition import LPData
from cuopt_server.utils.utils import build_lp_datamodel_from_json


def get_lp_json():
    return {
        "csr_constraint_matrix": {
            "offsets": [0, 2],
            "indices": [0, 1],
            "values": [1.0, 1.0],
        },
        "constraint_bounds": {"upper_bounds": [5000.0], "lower_bounds": [0.0]},
        "objective_data": {
            "coefficients": [1.2, 1.7],
            "scalability_factor": 1.0,
            "offset": 0.5,
        },
        "variable_bounds": {
            "upper_bounds": [3000.0, 5000.0],
            "lower_bounds": [0.0, 0.0],
        },
        "maximize": True,
        "variable_names": ["x", "y"],
        "solver_config": {"time_limit": 5, "iteration_limit": 100},
    }


def get_lp_data():
    return LPData.parse_obj(get_lp_json())


def test_create_data_model():
    warnings, data_model = conversion.create_data_model(get_lp_data())

    assert warnings == []
    assert data_model.get_constraint_matrix_values().tolist() == [1.0, 1.0]
    assert data_model.get_constraint_matrix_indices().tolist() == [0, 1]
    assert data_model.get_constraint_matrix_offsets().tolist() == [0, 2]
    assert data_model.get_constraint_lower_bounds().tolist() == [0.0]
    assert data_model.get_constraint_upper_bounds().tolist() == [5000.0]
    assert data_model.get_objective_coefficients().tolist() == [1.2, 1.7]
    assert data_model.get_objective_scaling_factor() == 1.0
    assert data_model.get_objective_offset() == 0.5
    assert data_model.get_variable_lower_bounds().tolist() == [0.0, 0.0]
    assert data_model.get_variable_upper_bounds().tolist() == [3000.0, 5000.0]
    assert data_model.get_variable_names() == ["x", "y"]


def test_create_solver_limits():
    warnings, solver_settings = conversion.create_solver(get_lp_data(), None)

    assert warnings == []
    assert float(solver_settings.get_parameter("time_limit")) == 5.0
    assert int(solver_settings.get_parameter("iteration_limit")) == 100


def test_create_solver_limits_clamped_by_environment(monkeypatch):
    monkeypatch.setenv("CUOPT_LP_TIME_LIMIT_SEC", "2")
    monkeypatch.setenv("CUOPT_LP_ITERATION_LIMIT", "10")

    _, solver_settings = conversion.create_solver(get_lp_data(), None)

    assert float(solver_settings.get_parameter("time_limit")) == 2.0
    assert int(solver_settings.get_parameter("iteration_limit")) == 10


def test_create_solver_warns_on_ignored_fields():
    data = get_lp_json()
    data["solver_config"]["user_problem_file"] = "problem.mps"
    data["solver_config"]["solution_file"] = "solution.txt"

    warnings, _ = conversion.create_solver(LPData.parse_obj(data), None)

    assert warnings == [
        conversion.ignored_warning("user_problem_file"),
        conversion.ignored_warning("solution_file"),
    ]


def test_build_lp_datamodel_from_json():
    data_model, solver_settings = build_lp_datamodel_from_json(get_lp_json())

    assert data_model.get_objective_coefficients().tolist() == [1.2, 1.7]
    assert float(solver_settings.get_parameter("time_limit")) == 5.0


def _http_warmstart():
    return {
        "current_primal_solution": [0.1, 0.2],
        "current_dual_solution": [0.3],
        "initial_primal_average": [0.1, 0.2],
        "initial_dual_average": [0.3],
        "current_ATY": [0.3],
        "sum_primal_solutions": [0.1, 0.2],
        "sum_dual_solutions": [0.3],
        "last_restart_duality_gap_primal_solution": [0.1, 0.2],
        "last_restart_duality_gap_dual_solution": [0.3],
        "initial_primal_weight": 1.0,
        "initial_step_size": 1.0,
        "total_pdlp_iterations": 1,
        "total_pdhg_iterations": 1,
        "last_candidate_kkt_score": 0.0,
        "last_restart_kkt_score": 0.0,
        "sum_solution_weight": 1.0,
        "iterations_since_last_restart": 0,
    }


def test_pdlp_http_warmstart_roundtrip():
    import numpy as np

    from cuopt.linear_programming.solution.solution import PDLPWarmStartData

    src = PDLPWarmStartData(
        np.array([0.1, 0.2]),
        np.array([0.3]),
        np.array([0.1, 0.2]),
        np.array([0.3]),
        np.array([0.3]),
        np.array([0.1, 0.2]),
        np.array([0.3]),
        np.array([0.1, 0.2]),
        np.array([0.3]),
        1.0,
        1.0,
        1,
        1,
        0.0,
        0.0,
        1.0,
        0,
    )
    http_dict = conversion.extract_pdlpwarmstart_data(src)
    # GET /warmstart returns the solver's float64 ndarrays, not lists
    assert isinstance(http_dict["current_primal_solution"], np.ndarray)
    assert http_dict["current_primal_solution"].dtype == np.float64
    restored = conversion.pdlp_from_http_warmstart(http_dict)
    assert list(restored.current_primal_solution) == [0.1, 0.2]
    assert restored.current_primal_solution.dtype == np.float64
    assert restored.initial_primal_weight == 1.0


def test_create_solver_applies_http_warmstart():
    _, solver_settings = conversion.create_solver(
        get_lp_data(), _http_warmstart()
    )
    ws = solver_settings.get_pdlp_warm_start_data()
    assert ws is not None
    assert list(ws.current_primal_solution) == [0.1, 0.2]
