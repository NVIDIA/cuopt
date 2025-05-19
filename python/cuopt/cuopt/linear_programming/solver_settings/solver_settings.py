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

from enum import IntEnum, auto

from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_ABSOLUTE_DUAL_TOLERANCE,
    CUOPT_ABSOLUTE_GAP_TOLERANCE,
    CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
    CUOPT_CROSSOVER,
    CUOPT_DUAL_INFEASIBLE_TOLERANCE,
    CUOPT_INFEASIBILITY_DETECTION,
    CUOPT_ITERATION_LIMIT,
    CUOPT_LOG_TO_CONSOLE,
    CUOPT_METHOD,
    CUOPT_MIP_ABSOLUTE_GAP,
    CUOPT_MIP_HEURISTICS_ONLY,
    CUOPT_MIP_INTEGRALITY_TOLERANCE,
    CUOPT_MIP_RELATIVE_GAP,
    CUOPT_MIP_SCALING,
    CUOPT_NUM_CPU_THREADS,
    CUOPT_PDLP_SOLVER_MODE,
    CUOPT_PRIMAL_INFEASIBLE_TOLERANCE,
    CUOPT_RELATIVE_DUAL_TOLERANCE,
    CUOPT_RELATIVE_GAP_TOLERANCE,
    CUOPT_RELATIVE_PRIMAL_TOLERANCE,
    CUOPT_TIME_LIMIT,
    get_solver_setting,
)


class SolverMethod(IntEnum):
    """
    Enum representing different methods to use for solving linear programs.
    """

    Concurrent = 0
    PDLP = auto()
    DualSimplex = auto()

    def __str__(self):
        """Convert the solver method to a string.

        Returns
        -------
            The string representation of the solver method.
        """
        return "%d" % self.value


class PDLPSolverMode(IntEnum):
    """
    Enum representing different solver modes to use in the
    `SolverSettings.set_pdlp_solver_mode` function.

    Attributes
    ----------
    Stable2
        Best overall mode from experiments; balances speed and convergence
        success. If you want to use the legacy version, use Stable1.
    Methodical1
        Takes slower individual steps, but fewer are needed to converge.
    Fast1
        Fastest mode, but with less success in convergence.

    Notes
    -----
    Default mode is Stable2.
    """

    Stable1 = 0
    Stable2 = auto()
    Methodical1 = auto()
    Fast1 = auto()

    def __str__(self):
        """Convert the solver mode to a string.

        Returns
        -------
        str
            The string representation of the solver mode.
        """
        return "%d" % self.value


class SolverSettings:
    def __init__(self):
        self.settings_dict = {}
        self.pdlp_warm_start_data = None
        self.mip_callbacks = []

    def to_base_type(self, value):
        """Convert a string to a base type.

        Parameters
        ----------
        value : str
            The value to convert.

        Returns
        -------
        value : float, int, bool, or str
            The converted value.
        """
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    value = value
        return value

    def get_parameter(self, name):
        """Get the value of a parameter used by cuOpt's LP/MIP solvers.

        Parameters
        ----------
        name : str
            The name of the parameter to get.

        Returns
        -------
        value : float, int, bool, or str
            The value of the parameter.

        Notes
        -----
        For a list of availabe parameters, their descriptions, default values,
        and acceptable ranges, see the cuOpt documentation `parameter.rst`.
        """
        if name in self.settings_dict:
            if isinstance(self.settings_dict[name], str):
                return self.to_base_type(self.settings_dict[name])
            else:
                return self.settings_dict[name]
        else:
            value = self.to_base_type(get_solver_setting(name))
            self.settings_dict[name] = value
            return value

    def set_parameter(self, name, value):
        """Set the value of a parameter used by cuOpt's LP/MIP solvers.

        Parameters
        ----------
        name : str
            The name of the parameter to set.
        value : str
            The value the parameter should take.

        For a list of availabe parameters, their descriptions, default values,
        and acceptable ranges, see the cuOpt documentation `parameter.rst`.
        """

        self.settings_dict[name] = value

    def set_optimality_tolerance(self, eps_optimal):
        """
        NOTE: Not supported for MILP, absolute is fixed to 1e-4,

        Set both absolute and relative tolerance on the primal feasibility,
        dual feasibility, and gap.
        Changing this value has a significant impact on accuracy and runtime.

        Optimality is computed as follows:

        dual_feasibility < absolute_dual_tolerance + relative_dual_tolerance
          * norm_objective_coefficient (l2_norm(c))
        primal_feasibility < absolute_primal_tolerance
          + relative_primal_tolerance * norm_constraint_bounds (l2_norm(b))
        duality_gap < absolute_gap_tolerance + relative_gap_tolerance
          * (abs(primal_objective) + abs(dual_objective))

        If all three conditions hold, optimality is reached.

        Parameters
        ----------
        eps_optimal : float64
            Tolerance to optimality
        Notes
        -----
        Default value is 1e-4.
        To set each absolute and relative tolerance, use the provided setters.
        """
        self.settings_dict["absolute_dual_tolerance"] = eps_optimal
        self.settings_dict["relative_dual_tolerance"] = eps_optimal
        self.settings_dict["absolute_primal_tolerance"] = eps_optimal
        self.settings_dict["relative_primal_tolerance"] = eps_optimal
        self.settings_dict["absolute_gap_tolerance"] = eps_optimal
        self.settings_dict["relative_gap_tolerance"] = eps_optimal

    def set_pdlp_warm_start_data(self, pdlp_warm_start_data):
        """
        Set the pdlp warm start data. This allows to restart PDLP with a
        previous solution context.

        This should be used when you solve a new problem which is similar to
        the previous one.

        Parameters
        ----------
        pdlp_warm_start_data : PDLPWarmStartData
            PDLP warm start data.

        Notes
        -----
        For now, the problem must have the same number of variables and
        constraints as the one found in the previous solution.

        Only supported solver modes are Stable2 and Fast1.

        Examples
        --------
        >>> solution = solver.Solve(first_problem, settings)
        >>> settings.set_pdlp_warm_start_data(
        >>>     solution.get_pdlp_warm_start_data()
        >>> )
        >>> solution = solver.Solve(second_problem, settings)
        """
        self.pdlp_warm_start_data = pdlp_warm_start_data

    def set_mip_callback(self, callback):
        """
        Note: Only supported for MILP

        Set the callback to receive incumbent solution.

        Parameters
        ----------
        callback : class for function callback
            Callback class that inherits from GetSolutionCallback
            or SetSolutionCallback.

        Examples
        --------
        >>> # Callback for incumbent solution
        >>> class CustomGetSolutionCallback(GetSolutionCallback):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.n_callbacks = 0
        >>>         self.solutions = []
        >>>
        >>>     def get_solution(self, solution, solution_cost):
        >>>         self.n_callbacks += 1
        >>>         assert len(solution) > 0
        >>>         assert len(solution_cost) == 1
        >>>
        >>>         self.solutions.append(
        >>>             {
        >>>                 "solution": solution.copy_to_host(),
        >>>                 "cost": solution_cost.copy_to_host()[0],
        >>>             }
        >>>         )
        >>>
        >>> class CustomSetSolutionCallback(SetSolutionCallback):
        >>>     def __init__(self, get_callback):
        >>>         super().__init__()
        >>>         self.n_callbacks = 0
        >>>         self.get_callback = get_callback
        >>>
        >>>     def set_solution(self, solution, solution_cost):
        >>>         self.n_callbacks += 1
        >>>         if self.get_callback.solutions:
        >>>             solution[:] =
        >>>             self.get_callback.solutions[-1]["solution"]
        >>>             solution_cost[0] = float(
        >>>                 self.get_callback.solutions[-1]["cost"]
        >>>             )
        >>>
        >>> get_callback = CustomGetSolutionCallback()
        >>> set_callback = CustomSetSolutionCallback(get_callback)
        >>> settings.set_mip_callback(get_callback)
        >>> settings.set_mip_callback(set_callback)
        """
        self.mip_callbacks.append(callback)

    def get_mip_callbacks(self):
        """
        Return callback class object
        """
        return self.mip_callbacks

    def get_pdlp_warm_start_data(self):
        """
        Returns the warm start data. See `set_pdlp_warm_start_data` for more
        details.

        Returns
        -------
        pdlp_warm_start_data

        Notes
        -----
        ...

        """
        return self.pdlp_warm_start_data

    def toDict(self):

        time_limit = self.get_parameter(CUOPT_TIME_LIMIT)
        if time_limit == float("inf"):
            time_limit = None

        solver_config = {
            "tolerances": {},
            "infeasibility_detection": self.get_parameter(
                CUOPT_INFEASIBILITY_DETECTION
            ),
            "time_limit": time_limit,
            "iteration_limit": self.get_parameter(CUOPT_ITERATION_LIMIT),
            "solver_mode": self.get_parameter(CUOPT_PDLP_SOLVER_MODE),
            "method": self.get_parameter(CUOPT_METHOD),
            "mip_scaling": self.get_parameter(CUOPT_MIP_SCALING),
            "heuristics_only": self.get_parameter(CUOPT_MIP_HEURISTICS_ONLY),
            "num_cpu_threads": self.get_parameter(CUOPT_NUM_CPU_THREADS),
            "crossover": self.get_parameter(CUOPT_CROSSOVER),
            "log_to_console": self.get_parameter(CUOPT_LOG_TO_CONSOLE),
        }
        solver_config["tolerances"]["absolute_dual"] = self.get_parameter(
            CUOPT_ABSOLUTE_DUAL_TOLERANCE
        )
        solver_config["tolerances"]["relative_dual"] = self.get_parameter(
            CUOPT_RELATIVE_DUAL_TOLERANCE
        )
        solver_config["tolerances"]["absolute_primal"] = self.get_parameter(
            CUOPT_ABSOLUTE_PRIMAL_TOLERANCE
        )
        solver_config["tolerances"]["relative_primal"] = self.get_parameter(
            CUOPT_RELATIVE_PRIMAL_TOLERANCE
        )
        solver_config["tolerances"]["absolute_gap"] = self.get_parameter(
            CUOPT_ABSOLUTE_GAP_TOLERANCE
        )
        solver_config["tolerances"]["relative_gap"] = self.get_parameter(
            CUOPT_RELATIVE_GAP_TOLERANCE
        )
        solver_config["tolerances"]["primal_infeasible"] = self.get_parameter(
            CUOPT_PRIMAL_INFEASIBLE_TOLERANCE
        )
        solver_config["tolerances"]["dual_infeasible"] = self.get_parameter(
            CUOPT_DUAL_INFEASIBLE_TOLERANCE
        )
        solver_config["tolerances"][
            "integrality_tolerance"
        ] = self.get_parameter(CUOPT_MIP_INTEGRALITY_TOLERANCE)
        solver_config["tolerances"]["absolute_mip_gap"] = self.get_parameter(
            CUOPT_MIP_ABSOLUTE_GAP
        )
        solver_config["tolerances"]["relative_mip_gap"] = self.get_parameter(
            CUOPT_MIP_RELATIVE_GAP
        )
        return solver_config
