# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for toDict / toDataModelAndSettings / toDictFromSolution."""

import copy
import json
import os
import tempfile
import zlib
from types import SimpleNamespace

import msgpack
import msgpack_numpy
import numpy as np
import pytest

from cuopt.linear_programming.io import (
    toDataModelAndSettings,
    toDict,
    toDictFromSolution,
)
from cuopt.linear_programming.solution.solution import (
    LPTerminationStatus,
    ProblemCategory,
)

LP_EXAMPLE = {
    "csr_constraint_matrix": {
        "offsets": [0, 2, 4],
        "indices": [0, 1, 0, 1],
        "values": [3.0, 4.0, 2.7, 10.1],
    },
    "constraint_bounds": {
        "upper_bounds": [5.4, 4.9],
        "lower_bounds": ["ninf", "ninf"],
    },
    "objective_data": {
        "coefficients": [0.2, 0.1],
        "scalability_factor": 1.0,
        "offset": 0.0,
    },
    "variable_bounds": {
        "upper_bounds": ["inf", "inf"],
        "lower_bounds": [0.0, 0.0],
    },
    "maximize": False,
    "variable_names": ["x", "y"],
    "solver_config": {"tolerances": {"optimality": 0.0001}, "time_limit": 5},
}


class _FakeSol:
    def get_termination_status(self):
        return LPTerminationStatus.Optimal

    def get_termination_reason(self):
        return "Optimal"

    def get_problem_category(self):
        return ProblemCategory.LP

    def get_primal_solution(self):
        return np.array([1.0, 2.0])

    def get_dual_solution(self):
        return np.array([0.5])

    def get_primal_objective(self):
        return 3.0

    def get_dual_objective(self):
        return 3.0

    def get_solve_time(self):
        return 0.01

    def get_solved_by(self):
        return SimpleNamespace(name="PDLP")

    def get_vars(self):
        return {"x": 1.0, "y": 2.0}

    def get_lp_stats(self):
        return {"nb_iterations": 1}

    def get_reduced_cost(self):
        return np.array([0.0, 0.0])

    def get_milp_stats(self):
        return None


def test_to_data_model_rejects_json_array_input():
    with pytest.raises(TypeError, match="Expected a cuOpt JSON object"):
        toDataModelAndSettings("[]")


def test_to_data_model_from_mapping():
    dm, settings = toDataModelAndSettings(copy.deepcopy(LP_EXAMPLE))
    assert len(dm.get_objective_coefficients()) == 2
    assert np.allclose(dm.get_objective_coefficients(), [0.2, 0.1])
    assert np.isinf(dm.get_variable_upper_bounds()).all()
    assert np.isneginf(dm.get_constraint_lower_bounds()).all()
    assert list(dm.get_variable_names()) == ["x", "y"]
    assert settings.get_parameter("time_limit") == 5
    assert settings.get_parameter("absolute_primal_tolerance") == 0.0001


def test_to_data_model_defaults_without_solver_config():
    payload = copy.deepcopy(LP_EXAMPLE)
    del payload["solver_config"]
    dm, settings = toDataModelAndSettings(payload)
    assert len(dm.get_objective_coefficients()) == 2
    assert settings.settings_dict == {}


def test_to_data_model_reads_mip_gap_from_nested_tolerances():
    payload = copy.deepcopy(LP_EXAMPLE)
    payload["solver_config"]["tolerances"]["mip_relative_gap"] = 0.01
    payload["solver_config"]["tolerances"]["mip_absolute_gap"] = 0.02
    _dm, settings = toDataModelAndSettings(payload)
    assert settings.get_parameter("mip_relative_gap") == 0.01
    assert settings.get_parameter("mip_absolute_gap") == 0.02


def test_to_data_model_coerces_boolean_solver_parameters_for_grpc():
    payload = copy.deepcopy(LP_EXAMPLE)
    payload["solver_config"]["log_to_console"] = False
    payload["solver_config"]["mip_scaling"] = True
    _dm, settings = toDataModelAndSettings(payload)
    assert settings.settings_dict["log_to_console"] == 0
    assert settings.settings_dict["mip_scaling"] == 1


def test_to_data_model_from_json_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as fh:
        json.dump(LP_EXAMPLE, fh)
        path = fh.name
    try:
        dm, _settings = toDataModelAndSettings(path)
        assert len(dm.get_objective_coefficients()) == 2
    finally:
        os.unlink(path)


def test_to_data_model_from_msgpack_file():
    msgpack_numpy.patch()
    payload = copy.deepcopy(LP_EXAMPLE)
    payload["objective_data"]["coefficients"] = np.array([0.2, 0.1])
    with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as fh:
        fh.write(msgpack.dumps(payload))
        path = fh.name
    try:
        dm, settings = toDataModelAndSettings(path)
        assert np.allclose(dm.get_objective_coefficients(), [0.2, 0.1])
        assert settings.get_parameter("time_limit") == 5
    finally:
        os.unlink(path)


def test_to_data_model_from_zlib_file():
    with tempfile.NamedTemporaryFile(suffix=".zlib", delete=False) as fh:
        fh.write(zlib.compress(json.dumps(LP_EXAMPLE).encode()))
        path = fh.name
    try:
        dm, settings = toDataModelAndSettings(path)
        assert np.allclose(dm.get_objective_coefficients(), [0.2, 0.1])
        assert settings.get_parameter("time_limit") == 5
    finally:
        os.unlink(path)


def test_to_dict_round_trip_json_true_and_false():
    """A DataModel survives toDict -> toDataModelAndSettings for every schema field."""
    payload = copy.deepcopy(LP_EXAMPLE)
    payload["initial_solution"] = {"primal": [0.1, 0.2], "dual": [0.0, 1.0]}
    dm, _settings = toDataModelAndSettings(payload)

    for as_json in (True, False):
        encoded = toDict(dm, json=as_json)
        assert "solver_config" not in encoded
        dm2, _settings2 = toDataModelAndSettings(encoded)
        assert np.allclose(
            dm.get_objective_coefficients(), dm2.get_objective_coefficients()
        )
        assert np.allclose(
            dm.get_variable_lower_bounds(), dm2.get_variable_lower_bounds()
        )
        assert np.isinf(dm2.get_variable_upper_bounds()).all()
        assert np.isneginf(dm2.get_constraint_lower_bounds()).all()
        assert list(dm2.get_variable_names()) == ["x", "y"]
        assert np.allclose(dm2.initial_primal_solution, [0.1, 0.2])
        assert np.allclose(dm2.initial_dual_solution, [0.0, 1.0])


def test_to_dict_emits_only_the_starts_that_are_set():
    payload = copy.deepcopy(LP_EXAMPLE)
    payload["initial_solution"] = {"primal": [0.1, 0.2]}
    dm, _settings = toDataModelAndSettings(payload)
    for as_json in (True, False):
        initial = toDict(dm, json=as_json)["initial_solution"]
        assert np.allclose(initial["primal"], [0.1, 0.2])
        assert "dual" not in initial


def test_to_dict_omits_initial_solution_when_model_has_none():
    dm, _settings = toDataModelAndSettings(copy.deepcopy(LP_EXAMPLE))
    assert "initial_solution" not in toDict(dm, json=True)
    assert "initial_solution" not in toDict(dm, json=False)


def test_to_dict_from_solution_envelope():
    body = toDictFromSolution(_FakeSol())
    assert body["reqId"] is None
    assert body["warnings"] is None
    assert body["response"]["total_solve_time"] is None
    solver = body["response"]["solver_response"]
    assert solver["status"] == "Optimal"
    sol = solver["solution"]
    assert sol["primal_objective"] == 3.0
    assert sol["primal_solution"] == [1.0, 2.0]
    assert sol["vars"] == {"x": 1.0, "y": 2.0}
    assert sol["solved_by"] == "PDLP"
    assert "pdlpwarmstart_data" not in sol
    assert sol["lp_statistics"] == {"nb_iterations": 1}
    assert sol["milp_statistics"] == {}
    assert body["notes"] == ["Optimal"]
