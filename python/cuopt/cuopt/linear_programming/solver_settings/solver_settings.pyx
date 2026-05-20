# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

"""Cython extension: LP/MIP ``SolverSettings`` backed by ``solver_settings_t``."""

from enum import IntEnum, auto

from cuopt.utilities import get_data_ptr

from libcpp.memory cimport unique_ptr
from libc.stdint cimport uintptr_t
from libcpp.string cimport string
from libcpp.vector cimport vector


def get_solver_setting(name):
    """Return the default string form of solver parameter *name* from a fresh C++ settings object."""
    cdef unique_ptr[solver_settings_t[int, double]] unique_solver_settings

    unique_solver_settings.reset(new solver_settings_t[int, double]())

    cdef solver_settings_t[int, double]* c_solver_settings = (
        unique_solver_settings.get()
    )
    return c_solver_settings.get_parameter_as_string(
        name.encode('utf-8')
    ).decode('utf-8')


cpdef get_solver_parameter_names():
    """Return all registered solver parameter names (same order as the C++ layer)."""
    cdef unique_ptr[solver_settings_t[int, double]] unique_solver_settings
    unique_solver_settings.reset(new solver_settings_t[int, double]())
    cdef solver_settings_t[int, double]* c_solver_settings = (
        unique_solver_settings.get()
    )
    cdef vector[string] parameter_names = c_solver_settings.get_parameter_names()

    cdef list py_parameter_names = []
    cdef size_t i
    for i in range(parameter_names.size()):
        # std::string -> Python str
        py_parameter_names.append(parameter_names[i].decode("utf-8"))
    return py_parameter_names


solver_params = get_solver_parameter_names()
for param in solver_params: globals()["CUOPT_"+param.upper()] = param


class SolverMethod(IntEnum):
    """
    Enum representing different methods to use for solving linear programs.
    """

    Concurrent = 0
    PDLP = auto()
    DualSimplex = auto()
    Barrier = auto()
    Unset = auto()

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


cdef class SolverSettings:
    def __init__(self):
        self.c_solver_settings.reset(new solver_settings_t[int, double]())
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

    def set_c_solver_settings(self):
        """Replay Python-side state into the C++ ``solver_settings_t`` object.

        Reset-replay invariant
        ----------------------
        ``Solve`` and ``BatchSolve`` call ``c_solver_settings.reset(new ...)``
        in ``solver_wrapper`` immediately before this method. Every solve therefore
        uses a **new** C++ settings object that is filled from Python, not from
        whatever was left in C++ after the previous solve.

        Source of truth on the Python object:

        * ``settings_dict`` — solver parameters (use :meth:`set_parameter` /
          :meth:`get_parameter`, not direct C++ mutation).
        * ``pdlp_warm_start_data`` — PDLP warm start (see :meth:`set_pdlp_warm_start_data`).
        * ``mip_callbacks`` — MIP callbacks (see :meth:`set_mip_callback`).

        Any change made only on ``c_solver_settings`` without updating these Python
        attributes is **discarded** on the next solve.

        :meth:`load_parameters_from_file` loads into C++ then mirrors every
        parameter back into ``settings_dict`` so :meth:`get_parameter` and the next
        replay stay consistent.
        """
        # All cdef declarations must precede other statements in this function.
        cdef solver_settings_t[int, double]* c_solver_settings
        cdef uintptr_t c_current_primal_solution
        cdef uintptr_t c_current_dual_solution
        cdef uintptr_t c_initial_primal_average
        cdef uintptr_t c_initial_dual_average
        cdef uintptr_t c_current_ATY
        cdef uintptr_t c_sum_primal_solutions
        cdef uintptr_t c_sum_dual_solutions
        cdef uintptr_t c_last_restart_duality_gap_primal_solution
        cdef uintptr_t c_last_restart_duality_gap_dual_solution

        c_solver_settings = self.c_solver_settings.get()

        for name, value in self.settings_dict.items():
            c_solver_settings.set_parameter_from_string(
                name.encode('utf-8'),
                str(value).encode('utf-8')
            )

        if self.get_pdlp_warm_start_data() is not None:
            warm_start_data = self.get_pdlp_warm_start_data()
            c_current_primal_solution = (
                get_data_ptr(
                    warm_start_data.current_primal_solution # noqa
                )
            )
            c_current_dual_solution = (
                get_data_ptr(
                    warm_start_data.current_dual_solution
                )
            )
            c_initial_primal_average = (
                get_data_ptr(
                    warm_start_data.initial_primal_average # noqa
                )
            )
            c_initial_dual_average = (
                get_data_ptr(
                    warm_start_data.initial_dual_average
                )
            )
            c_current_ATY = (
                get_data_ptr(
                    warm_start_data.current_ATY
                )
            )
            c_sum_primal_solutions = (
                get_data_ptr(
                    warm_start_data.sum_primal_solutions
                )
            )
            c_sum_dual_solutions = (
                get_data_ptr(
                    warm_start_data.sum_dual_solutions
                )
            )
            c_last_restart_duality_gap_primal_solution = (
                get_data_ptr(
                    warm_start_data.last_restart_duality_gap_primal_solution # noqa
                )
            )
            c_last_restart_duality_gap_dual_solution = (
                get_data_ptr(
                    warm_start_data.last_restart_duality_gap_dual_solution # noqa
                )
            )
            c_solver_settings.set_pdlp_warm_start_data(
                <const double *> c_current_primal_solution,
                <const double *> c_current_dual_solution,
                <const double *> c_initial_primal_average,
                <const double *> c_initial_dual_average,
                <const double *> c_current_ATY,
                <const double *> c_sum_primal_solutions,
                <const double *> c_sum_dual_solutions,
                <const double *> c_last_restart_duality_gap_primal_solution,
                <const double *> c_last_restart_duality_gap_dual_solution,
                warm_start_data.last_restart_duality_gap_primal_solution.shape[0], # Primal size # noqa
                warm_start_data.last_restart_duality_gap_dual_solution.shape[0], # Dual size # noqa
                warm_start_data.initial_primal_weight,
                warm_start_data.initial_step_size,
                warm_start_data.total_pdlp_iterations,
                warm_start_data.total_pdhg_iterations,
                warm_start_data.last_candidate_kkt_score,
                warm_start_data.last_restart_kkt_score,
                warm_start_data.sum_solution_weight,
                warm_start_data.iterations_since_last_restart # noqa
            )

    def dump_parameters_to_file(self, path, hyperparameters_only=True):
        """Apply ``settings_dict`` / warm start to C++, then dump parameters to *path*.

        Calls :meth:`set_c_solver_settings` then the C++ ``solver_settings_t::dump_parameters_to_file``.

        Parameters
        ----------
        path : str
            Output path (e.g. file path or ``/dev/stdout``).
        hyperparameters_only : bool, optional
            Forwarded to C++; when ``True``, dump hyperparameter subset only.

        Returns
        -------
        bool
            ``True`` if the C++ layer reports success.
        """
        self.set_c_solver_settings()
        cdef solver_settings_t[int, double]* c_ss = self.c_solver_settings.get()
        cdef string c_path = path.encode("utf-8")
        return c_ss.dump_parameters_to_file(c_path, hyperparameters_only)

    def load_parameters_from_file(self, path):
        """Load parameters from a cuOpt config file.

        Loads into the current C++ object, then refreshes :attr:`settings_dict`
        from C++ for each registered parameter name. That keeps
        :meth:`get_parameter` aligned with what :meth:`set_c_solver_settings` will
        replay on the next solve (see reset-replay invariant on
        :meth:`set_c_solver_settings`).

        Parameters
        ----------
        path : str
            Path to a parameter file with ``name = value`` lines (see C++
            ``solver_settings_t::load_parameters_from_file``).
        """
        cdef solver_settings_t[int, double]* c_ss = self.c_solver_settings.get()
        cdef string c_path = path.encode("utf-8")
        cdef string c_name
        cdef string c_val
        c_ss.load_parameters_from_file(c_path)
        for name in solver_params:
            c_name = name.encode("utf-8")
            c_val = c_ss.get_parameter_as_string(c_name)
            self.settings_dict[name] = self.to_base_type(c_val.decode("utf-8"))

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
