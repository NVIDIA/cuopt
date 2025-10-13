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

import numpy as np
import pytest

from cuopt import linear_programming
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_ABSOLUTE_DUAL_TOLERANCE,
    CUOPT_ABSOLUTE_GAP_TOLERANCE,
    CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
    CUOPT_AUGMENTED,
    CUOPT_BARRIER_DUAL_INITIAL_POINT,
    CUOPT_CUDSS_DETERMINISTIC,
    CUOPT_DUAL_INFEASIBLE_TOLERANCE,
    CUOPT_DUALIZE,
    CUOPT_ELIMINATE_DENSE_COLUMNS,
    CUOPT_FOLDING,
    CUOPT_METHOD,
    CUOPT_ORDERING,
    CUOPT_PRIMAL_INFEASIBLE_TOLERANCE,
    CUOPT_RELATIVE_DUAL_TOLERANCE,
    CUOPT_RELATIVE_GAP_TOLERANCE,
    CUOPT_RELATIVE_PRIMAL_TOLERANCE,
    CUOPT_TIME_LIMIT,
)
from cuopt.linear_programming.solver.solver_wrapper import LPTerminationStatus
from cuopt.linear_programming.solver_settings import SolverMethod, SolverSettings


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
def test_barrier_solver_options_python_api(
    folding,
    dualize,
    ordering,
    augmented,
    eliminate_dense,
    cudss_determ,
    dual_initial_point,
):
    """
    Test the barrier solver (method=Barrier) with various configuration options
    using the cuOpt Python API directly (not through the server).

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
    # Create a DataModel
    data_model = linear_programming.DataModel()

    # Set up the same problem as get_std_data_for_lp()
    # Minimize: 1.2*x + 1.7*y
    # Subject to: x + y <= 5000
    #             0 <= x <= 3000
    #             0 <= y <= 5000

    # Set CSR constraint matrix (1 constraint: x + y <= 5000)
    A_values = np.array([1.0, 1.0], dtype=np.float64)
    A_indices = np.array([0, 1], dtype=np.int32)
    A_offsets = np.array([0, 2], dtype=np.int32)
    data_model.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    # Set constraint bounds (right-hand side)
    constraint_lower_bounds = np.array([0.0], dtype=np.float64)
    constraint_upper_bounds = np.array([5000.0], dtype=np.float64)
    data_model.set_constraint_lower_bounds(constraint_lower_bounds)
    data_model.set_constraint_upper_bounds(constraint_upper_bounds)

    # Set objective coefficients
    c = np.array([1.2, 1.7], dtype=np.float64)
    data_model.set_objective_coefficients(c)

    # Set variable bounds
    variable_lower_bounds = np.array([0.0, 0.0], dtype=np.float64)
    variable_upper_bounds = np.array([3000.0, 5000.0], dtype=np.float64)
    data_model.set_variable_lower_bounds(variable_lower_bounds)
    data_model.set_variable_upper_bounds(variable_upper_bounds)

    # Set variable names
    variable_names = np.array(["x", "y"])
    data_model.set_variable_names(variable_names)

    # Minimize (default)
    data_model.set_maximize(False)

    # Create solver settings
    settings = SolverSettings()

    # Use barrier solver
    settings.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)

    # Set time limit
    settings.set_parameter(CUOPT_TIME_LIMIT, 5)

    # Set tolerances using parameter constants
    settings.set_parameter(CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 0.0001)
    settings.set_parameter(CUOPT_ABSOLUTE_DUAL_TOLERANCE, 0.0001)
    settings.set_parameter(CUOPT_ABSOLUTE_GAP_TOLERANCE, 0.0001)
    settings.set_parameter(CUOPT_RELATIVE_PRIMAL_TOLERANCE, 0.0001)
    settings.set_parameter(CUOPT_RELATIVE_DUAL_TOLERANCE, 0.0001)
    settings.set_parameter(CUOPT_RELATIVE_GAP_TOLERANCE, 0.0001)
    settings.set_parameter(CUOPT_PRIMAL_INFEASIBLE_TOLERANCE, 0.00000001)
    settings.set_parameter(CUOPT_DUAL_INFEASIBLE_TOLERANCE, 0.00000001)

    # Configure barrier solver options
    settings.set_parameter(CUOPT_FOLDING, folding)
    settings.set_parameter(CUOPT_DUALIZE, dualize)
    settings.set_parameter(CUOPT_ORDERING, ordering)
    settings.set_parameter(CUOPT_AUGMENTED, augmented)
    settings.set_parameter(CUOPT_ELIMINATE_DENSE_COLUMNS, eliminate_dense)
    settings.set_parameter(CUOPT_CUDSS_DETERMINISTIC, cudss_determ)
    settings.set_parameter(CUOPT_BARRIER_DUAL_INITIAL_POINT, dual_initial_point)

    print("\n=== Barrier Solver Test Configuration (Python API) ===")
    print(f"folding={folding}, dualize={dualize}, ordering={ordering}")
    print(f"augmented={augmented}, eliminate_dense={eliminate_dense}")
    print(f"cudss_deterministic={cudss_determ}")
    print(f"barrier_dual_initial_point={dual_initial_point}")

    # Solve the problem
    print("About to call Solve()...")
    solution = linear_programming.Solve(data_model, settings)
    print(f"Solve() returned, solution object: {solution}")
    print(f"Solution type: {type(solution)}")

    try:
        print("Getting termination status...")
        status = solution.get_termination_status()
        print(f"Termination Status: {status}")
    except Exception as e:
        print(f"ERROR getting termination status: {e}")
        raise

    try:
        print("Getting primal objective...")
        primal_obj = solution.get_primal_objective()
        print(f"Primal Objective: {primal_obj}")
    except Exception as e:
        print(f"ERROR getting primal objective: {e}")
        raise

    try:
        print("Getting dual objective...")
        dual_obj = solution.get_dual_objective()
        print(f"Dual Objective: {dual_obj}")
    except Exception as e:
        print(f"ERROR getting dual objective: {e}")
        raise

    try:
        print("Getting primal solution...")
        primal_sol = solution.get_primal_solution()
        print(f"Primal Solution: {primal_sol}")
    except Exception as e:
        print(f"ERROR getting primal solution: {e}")
        raise

    try:
        print("Getting dual solution...")
        dual_sol = solution.get_dual_solution()
        print(f"Dual Solution: {dual_sol}")
    except Exception as e:
        print(f"ERROR getting dual solution: {e}")
        raise

    try:
        print("Getting variables...")
        vars = solution.get_vars()
        print(f"Variables: {vars}")
    except Exception as e:
        print(f"ERROR getting variables: {e}")
        raise

    # Validate results
    assert (
        status == LPTerminationStatus.Optimal
    ), f"Expected Optimal status, got {status}"
    assert primal_sol is not None
    assert len(primal_sol) == 2
    assert dual_sol is not None
    assert primal_obj is not None

