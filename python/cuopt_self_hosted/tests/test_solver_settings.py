# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_sh_client import SolverMethod, ThinClientSolverSettings


def test_primal_solver_method_matches_native_value():
    assert SolverMethod.PrimalSimplex == 4


def test_settings_parameters_and_tolerances_go_to_the_request_dict():
    settings = ThinClientSolverSettings()
    settings.set_parameter("time_limit", 5.0)
    settings.set_optimality_tolerance(1e-6)

    assert settings.get_parameter("time_limit") == 5.0
    assert settings.get_parameter("unknown_parameter") is None

    config = settings.toDict()
    assert config["time_limit"] == 5.0
    assert config["tolerances"]["absolute_primal_tolerance"] == 1e-6
    assert config["tolerances"]["relative_gap_tolerance"] == 1e-6
