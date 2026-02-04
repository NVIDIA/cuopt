# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import cuopt_mps_parser
import pytest

from cuopt.linear_programming import solver, solver_settings
from cuopt.linear_programming.internals import (
    GetSolutionCallback,
    SetSolutionCallback,
)
from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT
from cuopt.linear_programming.solver.solver_wrapper import (
    MILPTerminationStatus,
)
from cuopt.utilities import utils

RAPIDS_DATASET_ROOT_DIR = os.getenv("RAPIDS_DATASET_ROOT_DIR")
if RAPIDS_DATASET_ROOT_DIR is None:
    RAPIDS_DATASET_ROOT_DIR = os.getcwd()
    RAPIDS_DATASET_ROOT_DIR = os.path.join(RAPIDS_DATASET_ROOT_DIR, "datasets")


def _run_incumbent_solver_callback(file_name, include_set_callback):
    # Callback for incumbent solution
    class CustomGetSolutionCallback(GetSolutionCallback):
        def __init__(self, user_data):
            super().__init__()
            self.n_callbacks = 0
            self.solutions = []
            self.user_data = user_data

        def get_solution(
            self, solution, solution_cost, solution_bound, user_data
        ):
            assert user_data is self.user_data
            self.n_callbacks += 1
            assert len(solution) > 0
            assert len(solution_cost) == 1
            assert len(solution_bound) == 1

            self.solutions.append(
                {
                    "solution": solution.tolist(),
                    "cost": float(solution_cost[0]),
                    "bound": float(solution_bound[0]),
                }
            )

    class CustomSetSolutionCallback(SetSolutionCallback):
        def __init__(self, get_callback, user_data):
            super().__init__()
            self.n_callbacks = 0
            self.get_callback = get_callback
            self.user_data = user_data

        def set_solution(
            self, solution, solution_cost, solution_bound, user_data
        ):
            assert user_data is self.user_data
            self.n_callbacks += 1
            assert len(solution_bound) == 1
            if self.get_callback.solutions:
                solution[:] = self.get_callback.solutions[-1]["solution"]
                solution_cost[0] = float(
                    self.get_callback.solutions[-1]["cost"]
                )

    user_data = {"source": "test_incumbent_solver_callback"}
    get_callback = CustomGetSolutionCallback(user_data)
    set_callback = (
        CustomSetSolutionCallback(get_callback, user_data)
        if include_set_callback
        else None
    )

    file_path = RAPIDS_DATASET_ROOT_DIR + file_name
    data_model_obj = cuopt_mps_parser.ParseMps(file_path)

    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_TIME_LIMIT, 10)
    settings.set_mip_callback(get_callback, user_data)
    if include_set_callback:
        settings.set_mip_callback(set_callback, user_data)
    solution = solver.Solve(data_model_obj, settings)

    assert get_callback.n_callbacks > 0
    if include_set_callback:
        assert set_callback.n_callbacks > 0
    assert (
        solution.get_termination_status()
        == MILPTerminationStatus.FeasibleFound
    )

    for sol in get_callback.solutions:
        utils.check_solution(
            data_model_obj, settings, sol["solution"], sol["cost"]
        )


@pytest.mark.parametrize(
    "file_name",
    [
        ("/mip/swath1.mps"),
        ("/mip/neos5-free-bound.mps"),
    ],
)
def test_incumbent_get_callback(file_name):
    _run_incumbent_solver_callback(file_name, include_set_callback=False)


@pytest.mark.parametrize(
    "file_name",
    [
        ("/mip/swath1.mps"),
        ("/mip/neos5-free-bound.mps"),
    ],
)
def test_incumbent_get_set_callback(file_name):
    _run_incumbent_solver_callback(file_name, include_set_callback=True)
