# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LP/MILP ``SolverConfig`` (passthrough / ``model_config``) validation."""

from cuopt_server.utils.linear_programming.data_definition import (
    LPData,
    SolverConfig,
    get_solver_config_value,
)


def test_solver_config_json_schema_allows_additional_properties():
    schema = SolverConfig.model_json_schema()
    assert schema.get("additionalProperties") is True
    assert "tolerances" in schema.get("properties", {})


def test_solver_config_accepts_passthrough_keys():
    sc = SolverConfig.model_validate(
        {
            "time_limit": 10.0,
            "mip_cut_passes": 4,
            "presolve": 1,
        }
    )
    dumped = sc.model_dump(mode="python")
    assert dumped["time_limit"] == 10.0
    assert dumped["mip_cut_passes"] == 4
    assert dumped["presolve"] == 1


def test_solver_config_tolerances_object_optional():
    sc = SolverConfig.model_validate({})
    assert sc.tolerances is None


def test_get_solver_config_value_nested_tolerances_first():
    sc = SolverConfig.model_validate(
        {
            "tolerances": {"absolute_primal_tolerance": 0.01},
        }
    )
    assert get_solver_config_value(sc, "absolute_primal_tolerance") == 0.01


def test_get_solver_config_value_top_level():
    sc = SolverConfig.model_validate({"time_limit": 5.0})
    assert get_solver_config_value(sc, "time_limit") == 5.0


def test_get_solver_config_value_nested_non_tolerance_name():
    """Keys like ``mip_absolute_gap`` may appear under ``tolerances`` in JSON."""
    sc = SolverConfig.model_validate(
        {"tolerances": {"mip_absolute_gap": 1e-9}}
    )
    assert get_solver_config_value(sc, "mip_absolute_gap") == 1e-9


def test_get_solver_config_value_top_level_overrides_when_not_in_tolerances():
    sc = SolverConfig.model_validate(
        {
            "time_limit": 9.0,
            "tolerances": {"absolute_primal_tolerance": 0.02},
        }
    )
    assert get_solver_config_value(sc, "time_limit") == 9.0
    assert get_solver_config_value(sc, "absolute_primal_tolerance") == 0.02


def test_lp_data_parses_with_passthrough_solver_config():
    payload = {
        "csr_constraint_matrix": {
            "offsets": [0, 2],
            "indices": [0, 1],
            "values": [1.0, 1.0],
        },
        "constraint_bounds": {"upper_bounds": [1.0], "lower_bounds": [0.0]},
        "objective_data": {
            "coefficients": [1.0, 1.0],
            "scalability_factor": 1.0,
            "offset": 0.0,
        },
        "variable_bounds": {
            "upper_bounds": [1.0, 1.0],
            "lower_bounds": [0.0, 0.0],
        },
        "solver_config": {
            "time_limit": 1.0,
            "mip_scaling": False,
            "tolerances": {"optimality": 1e-4},
        },
    }
    data = LPData.model_validate(payload)
    assert get_solver_config_value(data.solver_config, "time_limit") == 1.0
    assert get_solver_config_value(data.solver_config, "mip_scaling") is False
    assert get_solver_config_value(data.solver_config, "optimality") == 1e-4
