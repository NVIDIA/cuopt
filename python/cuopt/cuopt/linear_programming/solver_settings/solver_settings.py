# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, auto

from cuopt.linear_programming.solver.solver_parameters import (
    solver_params,
    get_solver_setting,
)


class SolverMethod(IntEnum):
    """
    Enum representing different methods to use for solving linear programs.
    """

    Concurrent = 0
    PDLP = auto()
    DualSimplex = auto()
    Barrier = auto()

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
    Stable3
        Best overall mode from experiments; balances speed and convergence
        success. If you want to use the legacy version, use Stable1.
    Methodical1
        Takes slower individual steps, but fewer are needed to converge.
    Fast1
        Fastest mode, but with less success in convergence.

    Notes
    -----
    Default mode is Stable3.
    """

    Stable1 = 0
    Stable2 = auto()
    Methodical1 = auto()
    Fast1 = auto()
    Stable3 = auto()

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
        if name not in solver_params:
            raise ValueError("Invalid parameter. Please check documentation")
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
        if name not in solver_params:
            raise ValueError("Invalid parameter. Please check documentation")
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
        for param in solver_params:
            if param.endswith("tolerance"):
                if not param.startswith("mip") and "infeasible" not in param:
                    self.settings_dict[param] = eps_optimal

    def set_pdlp_warm_start_data(self, pdlp_warm_start_data):
        """
        Set the pdlp warm start data. This allows to restart PDLP with a
        previous solution context.

        This should be used when you solve a new problem which is similar to
        the previous one.

        Parameters
        ----------
        pdlp_warm_start_data : PDLPWarmStartData
            PDLP warm start data obtained from a previous solve.
            Refer :py:meth:`cuopt.linear_programming.problem.Problem.getWarmstartData`  # noqa

        Notes
        -----
        For now, the problem must have the same number of variables and
        constraints as the one found in the previous solution.

        Only supported solver modes are Stable2 and Fast1.

        Examples
        --------
        >>> settings.set_pdlp_warm_start_data(pdlp_warm_start_data)
        """
        self.pdlp_warm_start_data = pdlp_warm_start_data

    def set_mip_callback(self, callback, user_data):
        """
        Note: Only supported for MILP

        Set the callback to receive incumbent solution.

        Parameters
        ----------
        callback : class for function callback
            Callback class that inherits from GetSolutionCallback
            or SetSolutionCallback.
        user_data : object
            User context passed to the callback.

        Notes
        -----
        Registering a SetSolutionCallback disables presolve.

        Examples
        --------
        >>> # Callback for incumbent solution
        >>> class CustomGetSolutionCallback(GetSolutionCallback):
        >>>     def __init__(self, user_data):
        >>>         super().__init__()
        >>>         self.n_callbacks = 0
        >>>         self.solutions = []
        >>>         self.user_data = user_data
        >>>
        >>>     def get_solution(
        >>>         self, solution, solution_cost, solution_bound, user_data
        >>>     ):
        >>>         assert user_data is self.user_data
        >>>         self.n_callbacks += 1
        >>>         assert len(solution) > 0
        >>>         assert len(solution_cost) == 1
        >>>         assert len(solution_bound) == 1
        >>>
        >>>         self.solutions.append(
        >>>             {
        >>>                 "solution": solution.tolist(),
        >>>                 "cost": float(solution_cost[0]),
        >>>                 "bound": float(solution_bound[0]),
        >>>             }
        >>>         )
        >>>
        >>> class CustomSetSolutionCallback(SetSolutionCallback):
        >>>     def __init__(self, get_callback, user_data):
        >>>         super().__init__()
        >>>         self.n_callbacks = 0
        >>>         self.get_callback = get_callback
        >>>         self.user_data = user_data
        >>>
        >>>     def set_solution(
        >>>         self, solution, solution_cost, solution_bound, user_data
        >>>     ):
        >>>         assert user_data is self.user_data
        >>>         self.n_callbacks += 1
        >>>         assert len(solution_bound) == 1
        >>>         if self.get_callback.solutions:
        >>>             solution[:] =
        >>>             self.get_callback.solutions[-1]["solution"]
        >>>             solution_cost[0] = float(
        >>>                 self.get_callback.solutions[-1]["cost"]
        >>>             )
        >>>
        >>> user_data = {"source": "example"}
        >>> get_callback = CustomGetSolutionCallback(user_data)
        >>> set_callback = CustomSetSolutionCallback(get_callback, user_data)
        >>> settings.set_mip_callback(get_callback, user_data)
        >>> settings.set_mip_callback(set_callback, user_data)
        """
        if callback is not None:
            callback.user_data = user_data
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
        pdlp_warm_start_data:

        """
        return self.pdlp_warm_start_data

    def toDict(self):
        solver_config = {}
        solver_config["tolerances"] = {}
        for param in solver_params:
            if param.endswith("tolerance"):
                solver_config["tolerances"][param] = self.get_parameter(param)
            else:
                param_value = self.get_parameter(param)
                if param_value == float("inf"):
                    param_value = None
                solver_config[param] = param_value

        return solver_config
