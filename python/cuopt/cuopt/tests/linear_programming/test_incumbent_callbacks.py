# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


@pytest.mark.parametrize(
    "file_name",
    [
        ("/mip/swath1.mps"),
        ("/mip/neos5-free-bound.mps"),
    ],
)
def test_incumbent_solver_callback(file_name):

    # Callback for incumbent solution
    class CustomGetSolutionCallback(GetSolutionCallback):
        def __init__(self):
            super().__init__()
            self.n_callbacks = 0
            self.solutions = []

        def get_solution(self, solution, solution_cost):

            self.n_callbacks += 1
            assert len(solution) > 0
            assert len(solution_cost) == 1

            self.solutions.append(
                {
                    "solution": solution.copy_to_host(),
                    "cost": solution_cost.copy_to_host()[0],
                }
            )

    class CustomSetSolutionCallback(SetSolutionCallback):
        def __init__(self, get_callback):
            super().__init__()
            self.n_callbacks = 0
            self.get_callback = get_callback

        def set_solution(self, solution, solution_cost):
            self.n_callbacks += 1
            if self.get_callback.solutions:
                solution[:] = self.get_callback.solutions[-1]["solution"]
                solution_cost[0] = float(
                    self.get_callback.solutions[-1]["cost"]
                )

    get_callback = CustomGetSolutionCallback()
    set_callback = CustomSetSolutionCallback(get_callback)

    file_path = RAPIDS_DATASET_ROOT_DIR + file_name
    data_model_obj = cuopt_mps_parser.ParseMps(file_path)

    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_TIME_LIMIT, 10)
    settings.set_mip_callback(get_callback)
    settings.set_mip_callback(set_callback)
    solution = solver.Solve(data_model_obj, settings)

    assert get_callback.n_callbacks > 0
    assert set_callback.n_callbacks > 0
    assert (
        solution.get_termination_status()
        == MILPTerminationStatus.FeasibleFound
    )

    for sol in get_callback.solutions:
        utils.check_solution(
            data_model_obj, settings, sol["solution"], sol["cost"]
        )
