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

from enum import IntEnum, auto


# Note these classes are only used on the thin client side.
# They are duplicates of the classes in
# cuopt.linear_programming.solver_settings.solver_settings
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


# Note these classes are only used on the thin client side.
# They are duplicates of the classes in
# cuopt.linear_programming.solver_settings.solver_settings
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


class ThinClientSolverSettings:
    def __init__(self):
        self.parameter_dict = {}

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
        self.parameter_dict["absolute_dual_tolerance"] = eps_optimal
        self.parameter_dict["relative_dual_tolerance"] = eps_optimal
        self.parameter_dict["absolute_primal_tolerance"] = eps_optimal
        self.parameter_dict["relative_primal_tolerance"] = eps_optimal
        self.parameter_dict["absolute_gap_tolerance"] = eps_optimal
        self.parameter_dict["relative_gap_tolerance"] = eps_optimal

    def set_parameter(self, name, value):
        """
        Set a parameter for the solver.

        Parameters
        ----------
        name : str
            The name of the parameter to set.
        value : object
            The value to set the parameter to.

        Notes
        -----
        For a list of available parameters, see the `settings.rst` file
        in the cuOpt documentation.
        """
        self.parameter_dict[name] = value

    def get_parameter(self, name):
        """
        Get a parameter for the solver.

        Parameters
        ----------
        name : str
            The name of the parameter to get.

        Returns
        -------
        object
            The value of the parameter.

        Notes
        -----
        For a list of available parameters, see the `settings.rst` file
        in the cuOpt documentation.
        """
        if name in self.parameter_dict:
            return self.parameter_dict[name]
        else:
            return None

    def toDict(self):
        solver_config = {
            "tolerances": {},
        }

        # Grab everything that is not a tolerance
        for key in self.parameter_dict:
            if "tolerance" not in key:
                solver_config[key] = self.parameter_dict[key]
        # Handle tolerance seperately
        if "absolute_dual_tolerance" in self.parameter_dict:
            solver_config["tolerances"]["absolute_dual"] = self.parameter_dict[
                "absolute_dual_tolerance"
            ]
        if "relative_dual_tolerance" in self.parameter_dict:
            solver_config["tolerances"]["relative_dual"] = self.parameter_dict[
                "relative_dual_tolerance"
            ]
        if "absolute_primal_tolerance" in self.parameter_dict:
            solver_config["tolerances"][
                "absolute_primal"
            ] = self.parameter_dict["absolute_primal_tolerance"]
        if "relative_primal_tolerance" in self.parameter_dict:
            solver_config["tolerances"][
                "relative_primal"
            ] = self.parameter_dict["relative_primal_tolerance"]
        if "absolute_gap_tolerance" in self.parameter_dict:
            solver_config["tolerances"]["absolute_gap"] = self.parameter_dict[
                "absolute_gap_tolerance"
            ]
        if "relative_gap_tolerance" in self.parameter_dict:
            solver_config["tolerances"]["relative_gap"] = self.parameter_dict[
                "relative_gap_tolerance"
            ]
        if "primal_infeasible_tolerance" in self.parameter_dict:
            solver_config["tolerances"][
                "primal_infeasible"
            ] = self.parameter_dict["primal_infeasible_tolerance"]
        if "dual_infeasible_tolerance" in self.parameter_dict:
            solver_config["tolerances"][
                "dual_infeasible"
            ] = self.parameter_dict["dual_infeasible_tolerance"]
        if "integrality_tolerance" in self.parameter_dict:
            solver_config["tolerances"][
                "integrality_tolerance"
            ] = self.parameter_dict["integrality_tolerance"]
        if "absolute_mip_gap" in self.parameter_dict:
            solver_config["tolerances"][
                "absolute_mip_gap"
            ] = self.parameter_dict["absolute_mip_gap"]
        if "relative_mip_gap" in self.parameter_dict:
            solver_config["tolerances"][
                "relative_mip_gap"
            ] = self.parameter_dict["relative_mip_gap"]
        return solver_config
