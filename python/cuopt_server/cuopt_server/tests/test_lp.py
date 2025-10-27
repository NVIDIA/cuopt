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

import pytest

from cuopt.linear_programming.solver.solver_wrapper import (
    LPTerminationStatus,
    MILPTerminationStatus,
    ProblemCategory,
)

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient, get_lp

client = RequestClient()


def validate_lp_result(
    res,
    expected_termination_status,
):
    sol = res["solution"]
    assert sol["problem_category"] == ProblemCategory.LP.name
    assert res["status"] == expected_termination_status
    assert len(sol["lp_statistics"]) > 1

    # MILP related values should be None or empty
    assert len(sol["milp_statistics"].keys()) == 0


def validate_milp_result(
    res,
    expected_termination_status,
):
    sol = res["solution"]
    assert sol["problem_category"] in (
        ProblemCategory.MIP.name,
        ProblemCategory.IP.name,
    )
    assert res["status"] == expected_termination_status
    assert len(sol["milp_statistics"].keys()) > 0

    # LP related values should be None or empty
    assert sol["dual_solution"] is None
    assert sol["dual_objective"] is None
    assert sol["reduced_cost"] is None
    assert len(sol["lp_statistics"]) == 0


def get_std_data_for_lp():

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
            "offset": 0.0,
        },
        "variable_bounds": {
            "upper_bounds": [3000.0, 5000.0],
            "lower_bounds": [0.0, 0.0],
        },
        "maximize": False,
        "variable_names": ["x", "y"],
        "solver_config": {
            "time_limit": 5,
            "tolerances": {
                "optimality": 0.0001,
                "absolute_primal": 0.0001,
                "absolute_dual": 0.0001,
                "absolute_gap": 0.0001,
                "relative_primal": 0.0001,
                "relative_dual": 0.0001,
                "relative_gap": 0.0001,
                "primal_infeasible": 0.00000001,
                "dual_infeasible": 0.00000001,
                "integrality_tolerance": 0.00001,
            },
        },
    }


def get_std_data_for_milp():

    data = get_std_data_for_lp()
    data["variable_types"] = ["I", "C"]
    data["maximize"] = True
    data["solver_config"]["mip_scaling"] = False
    return data


def test_sample_lp(cuoptproc):  # noqa

    res = get_lp(client, get_std_data_for_lp())

    assert res.status_code == 200

    print(res.json())
    validate_lp_result(
        res.json()["response"]["solver_response"],
        LPTerminationStatus.Optimal.name,
    )


@pytest.mark.parametrize(
    "maximize, scaling, expected_status, heuristics_only",
    [
        (True, True, MILPTerminationStatus.Optimal.name, True),
        (False, True, MILPTerminationStatus.Optimal.name, False),
        (True, False, MILPTerminationStatus.Optimal.name, True),
        (False, False, MILPTerminationStatus.Optimal.name, False),
    ],
)
def test_sample_milp(
    cuoptproc, maximize, scaling, expected_status, heuristics_only  # noqa
):

    data = get_std_data_for_milp()
    data["maximize"] = maximize
    data["solver_config"]["mip_scaling"] = scaling
    data["solver_config"]["heuristics_only"] = heuristics_only
    data["solver_config"]["num_cpu_threads"] = 4
    res = get_lp(client, data)

    assert res.status_code == 200

    print(res.json())
    validate_milp_result(
        res.json()["response"]["solver_response"],
        expected_status,
    )


@pytest.mark.skip(
    reason="Enable barrier solver options when issue "
    "https://github.com/NVIDIA/cuopt/issues/519 is resolved",
)
@pytest.mark.parametrize(
    "folding, dualize, ordering, augmented, eliminate_dense, cudss_determ, "
    "dual_initial_point",
    [
        # Test automatic settings (default)
        (-1, -1, -1, -1, True, False, -1),
        # Test folding off, no dualization, cuDSS default ordering, ADAT system
        (0, 0, 0, 0, True, False, 0),
        # Test folding on, force dualization, AMD ordering, augmented system
        (1, 1, 1, 1, True, True, 1),
        # Test mixed settings: automatic folding, no dualize, AMD, augmented
        (-1, 0, 1, 1, False, False, 0),
        # Test no folding, automatic dualize, cuDSS default, ADAT
        (0, -1, 0, 0, True, True, -1),
        # Test dual initial point with Lustig-Marsten-Shanno
        (-1, -1, -1, -1, True, False, 0),
        # Test dual initial point with least squares
        (-1, -1, -1, 1, True, False, 1),
    ],
)
def test_barrier_solver_options(
    cuoptproc,  # noqa
    folding,
    dualize,
    ordering,
    augmented,
    eliminate_dense,
    cudss_determ,
    dual_initial_point,
):
    """
    Test the barrier solver (method=3) with various configuration options:
    - folding: (-1) automatic, (0) off, (1) on
    - dualize: (-1) automatic, (0) don't dualize, (1) force dualize
    - ordering: (-1) automatic, (0) cuDSS default, (1) AMD
    - augmented: (-1) automatic, (0) ADAT, (1) augmented system
    - eliminate_dense_columns: True to eliminate, False to not
    - cudss_deterministic: True for deterministic, False for
      nondeterministic
    - barrier_dual_initial_point: (-1) automatic, (0) Lustig-Marsten-Shanno,
      (1) dual least squares
    """
    data = get_std_data_for_lp()

    # Use barrier solver (method=3)
    data["solver_config"]["method"] = 3

    # Configure barrier solver options
    data["solver_config"]["folding"] = folding
    data["solver_config"]["dualize"] = dualize
    data["solver_config"]["ordering"] = ordering
    data["solver_config"]["augmented"] = augmented
    data["solver_config"]["eliminate_dense_columns"] = eliminate_dense
    data["solver_config"]["cudss_deterministic"] = cudss_determ
    data["solver_config"]["barrier_dual_initial_point"] = dual_initial_point

    res = get_lp(client, data)

    assert res.status_code == 200

    print("\n=== Barrier Solver Test Configuration ===")
    print(f"folding={folding}, dualize={dualize}, ordering={ordering}")
    print(f"augmented={augmented}, eliminate_dense={eliminate_dense}")
    print(f"cudss_deterministic={cudss_determ}")
    print(f"barrier_dual_initial_point={dual_initial_point}")
    print(res.json())

    validate_lp_result(
        res.json()["response"]["solver_response"],
        LPTerminationStatus.Optimal.name,
    )
