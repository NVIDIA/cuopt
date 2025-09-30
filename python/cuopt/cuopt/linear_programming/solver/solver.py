# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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
import time

from cuopt.linear_programming.solver import solver_wrapper
from cuopt.linear_programming.solver_settings import SolverSettings
from cuopt.utilities import catch_cuopt_exception


@catch_cuopt_exception
def Solve(data_model, solver_settings=None):
    """
    Solve the Linear Program passed as input and returns the solution.

    Data Model object can be construed through setters
    (see linear_programming.DataModel class) or through a MPS file
    (see cuopt_mps_parser.ParseMps function)


    Notes
    -----
    Both primal and dual solutions are zero-initialized.
    For custom initialization, please refer to `set_initial_primal_solution()`
    and `set_initial_dual_solution()` methods.
    For more details on the Solution object, see linear_programming.Solution.

    Parameters
    ----------
    data_model : DataModel
        Data model containing a representation of a linear program on standard
        form.
    solver_settings: SolverSettings
        Settings to configure solver configurations.
        By default, it uses default solver settings to solve.

    Returns
    -------
    solution: Solution
        Solution object containing both primal and dual solutions, objectives
        and more statistics about the result.

    Examples
    --------
    >>> from cuopt import linear_programming
    >>> from cuopt.linear_programming.solver_settings import PDLPSolverMode
    >>> from cuopt.linear_programming.solver.solver_parameters import *
    >>>
    >>> data_model = linear_programming.DataModel()
    >>>
    >>> # Set all required fields
    >>> data_model.data_model.set_csr_constraint_matrix(...)
    >>>
    >>> # Build a solver setting object
    >>> settings = linear_programming.SolverSettings()
    >>> # Lower the accuracy from 1e-4 to 1e-2 for a faster result
    >>> settings.set_optimality_tolerance(1e-2)
    >>> # Also change the solver mode to Fast1 which can be faster
    >>> settings.set_parameter(CUOPT_PDLP_SOLVER_MODE, PDLPSolverMode.Fast1)
    >>>
    >>> # Call solve
    >>> solution = linear_programming.Solve(data_model, settings)
    >>>
    >>> # Print solution
    >>> print(solution.get_primal_solution())
    >>> # Print the primal objective
    >>> print(solution.get_primal_objective())
    >>> # Print the value of one specific variable
    >>> print(solution.get_vars()["var_name"])
    """

    emit_stamps = os.environ.get("CUOPT_EXTRA_TIMESTAMPS", False) in (
        True,
        "True",
        "true",
    )
    if emit_stamps:
        print(f"CUOPT_SOLVE_START: {time.time()}")

    if solver_settings is None:
        solver_settings = SolverSettings()

    def is_mip(var_types):
        if len(var_types) == 0:
            return False
        # Check if all types are the same (fast check)
        if len(set(map(type, var_types))) == 1:
            # Homogeneous - use appropriate check
            if isinstance(var_types[0], bytes):
                return b"I" in var_types
            else:
                return "I" in var_types
        else:
            # Mixed types - fallback to comprehensive check
            return any(vt == "I" or vt == b"I" for vt in var_types)

    s = solver_wrapper.Solve(
        data_model,
        solver_settings,
        mip=is_mip(data_model.get_variable_types()),
    )
    if emit_stamps:
        print(f"CUOPT_SOLVE_RETURN: {time.time()}")
    return s


@catch_cuopt_exception
def BatchSolve(data_model_list, solver_settings=None):
    """
    Solve the list of Linear Programs passed as input and returns the solutions
    and total solve time.

    Data Model objects can be construed through setters
    (see linear_programming.DataModel class) or through a MPS file
    (see cuopt_mps_parser.ParseMps function)


    Notes
    -----
    Solving in batch is usually faster than solving one by one.
    The total time to solve the whole batch on the engine side is returned as
    summing up the solutions `get_solve_time` would be incorrect as they are
    solved together in parallel, overlapping multiple solve.
    Both primal and dual solutions are zero-initialized.
    For custom initialization, please refer to `set_initial_primal_solution()`
    and `set_initial_dual_solution()` methods.
    For more details on the Solution object, see linear_programming.Solution.

    Parameters
    ----------
    data_models : list of DataModel
        Data models containing a representation of a linear program on standard
        form.
    solver_settings: SolverSettings
        Settings to configure solver configurations.
        By default, it uses default solver settings to solve.

    Returns
    -------
    solution: list of Solution
        List of Solution objects containing both primal and dual solutions,
        objectives and more statistics about the result.
    solve_time: double
        The engine solve time for the whole batch in seconds as a float64.

    Examples
    --------
    >>> from cuopt import linear_programming
    >>> from cuopt.linear_programming.solver_settings import PDLPSolverMode
    >>> from cuopt.linear_programming.solver.solver_parameters import *
    >>> import cuopt_mps_parser
    >>>
    >>> data_models = []
    >>> for i in range(...):
    >>>     data_models.append(cuopt_mps_parser.ParseMps(...))
    >>>
    >>> # Build a solver setting object
    >>> settings = linear_programming.SolverSettings()
    >>> # Lower the accuracy from 1e-4 to 1e-2 for a faster result
    >>> settings.set_optimality_tolerance(1e-2)
    >>> # Also change the solver mode to PDLPSolverMode.Fast1
    >>> # which can be faster
    >>> settings.set_parameter(CUOPT_PDLP_SOLVER_MODE, PDLPSolverMode.Fast1)
    >>>
    >>> # Call solve
    >>> solutions, solve_time = linear_programming.BatchSolve(
    >>>                            data_models,
    >>>                            settings
    >>>                         )
    >>>
    >>> # Print total engine solve time
    >>> print("Total engine solve time: ", solve_time)
    >>> # Print solutions
    >>> for solution in solutions:
    >>>     print(solution.get_primal_solution())
    >>>     # Print the primal objective
    >>>     print(solution.get_primal_objective())
    >>>     # Print the value of one specific variable
    >>>     print(solution.get_vars()["var_name"])
    """
    if solver_settings is None:
        solver_settings = SolverSettings()

    return solver_wrapper.BatchSolve(data_model_list, solver_settings)
