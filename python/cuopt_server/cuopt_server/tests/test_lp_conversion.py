# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_server.utils.linear_programming import conversion
from cuopt_server.utils.linear_programming import solver as lp_solver
from cuopt_server.utils.linear_programming.data_definition import LPData
from cuopt_server.utils.utils import build_lp_datamodel_from_json

CONVERSION_MODULE = "cuopt_server.utils.linear_programming.conversion"
SOLVER_MODULE = "cuopt_server.utils.linear_programming.solver"

CONVERSION_NAMES = [
    "create_data_model",
    "create_solver",
    "ignored_warning",
]

SOLVER_OWNED_NAMES = [
    "dep_warning",
    "warn_on_objectives",
]


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


def test_conversion_module_defines_helpers():
    # conversion.py owns only the helpers used by proxy conversion
    for name in CONVERSION_NAMES:
        func = getattr(conversion, name)
        assert func.__module__ == CONVERSION_MODULE
    for name in SOLVER_OWNED_NAMES:
        assert not hasattr(conversion, name)


def test_solver_reexports_are_not_duplicates():
    # solver.py must re-export conversion helpers, not redefine them
    for name in CONVERSION_NAMES:
        assert getattr(lp_solver, name) is getattr(conversion, name)


def test_solver_keeps_legacy_warning_helpers():
    # dep_warning / warn_on_objectives stay defined in solver.py
    for name in SOLVER_OWNED_NAMES:
        func = getattr(lp_solver, name)
        assert func.__module__ == SOLVER_MODULE


def test_solver_keeps_solve_side():
    # solve, callbacks and exception mapping stay in solver.py
    for name in [
        "solve",
        "get_solver_exception_type",
        "CustomGetSolutionCallback",
        "CustomSetSolutionCallback",
    ]:
        assert hasattr(lp_solver, name)
        assert not hasattr(conversion, name)


def test_server_utils_uses_conversion():
    from cuopt_server.utils import utils

    assert utils.lp_create_data_model is conversion.create_data_model
    assert utils.lp_create_solver is conversion.create_solver


def test_warning_helpers():
    assert "ignored" in conversion.ignored_warning("solution_file")
    assert "deprecated" in lp_solver.dep_warning("time_limit")
    assert lp_solver.warn_on_objectives("cfg") == ([], "cfg")


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
