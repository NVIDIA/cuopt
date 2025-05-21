# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
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


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from pylibraft.common.handle cimport *

from datetime import date, datetime

from dateutil.relativedelta import relativedelta

from cuopt.utilities import type_cast

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, strcpy, strlen
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.device_buffer cimport DeviceBuffer

from cuopt.linear_programming.data_model.data_model cimport data_model_view_t
from cuopt.linear_programming.data_model.data_model_wrapper cimport DataModel
from cuopt.linear_programming.solver.solver cimport (
    call_batch_solve,
    call_solve,
    error_type_t,
    mip_termination_status_t,
    pdlp_solver_mode_t,
    pdlp_termination_status_t,
    problem_category_t,
    solver_ret_t,
    solver_settings_t,
)

import math
import sys
import warnings
from enum import IntEnum

import cupy as cp
import numpy as np
from numba import cuda

import cudf
from cudf.core.buffer import as_buffer

from cuopt.linear_programming.solver.solver_parameters import CUOPT_LOG_FILE
from cuopt.linear_programming.solver_settings.solver_settings import (
    PDLPSolverMode,
    SolverSettings,
)
from cuopt.utilities import InputValidationError


cdef extern from "cuopt/linear_programming/utilities/internals.hpp" namespace "cuopt::internals": # noqa
    cdef cppclass base_solution_callback_t


class MILPTerminationStatus(IntEnum):
    NoTermination = mip_termination_status_t.NoTermination
    Optimal = mip_termination_status_t.Optimal
    FeasibleFound = mip_termination_status_t.FeasibleFound
    Infeasible = mip_termination_status_t.Infeasible
    Unbounded = mip_termination_status_t.Unbounded
    TimeLimit = mip_termination_status_t.TimeLimit


class LPTerminationStatus(IntEnum):
    NoTermination = pdlp_termination_status_t.NoTermination
    NumericalError = pdlp_termination_status_t.NumericalError
    Optimal = pdlp_termination_status_t.Optimal
    PrimalInfeasible = pdlp_termination_status_t.PrimalInfeasible
    DualInfeasible = pdlp_termination_status_t.DualInfeasible
    IterationLimit = pdlp_termination_status_t.IterationLimit
    TimeLimit = pdlp_termination_status_t.TimeLimit
    PrimalFeasible = pdlp_termination_status_t.PrimalFeasible


class ErrorStatus(IntEnum):
    Success = error_type_t.Success
    ValidationError = error_type_t.ValidationError
    OutOfMemoryError = error_type_t.OutOfMemoryError
    RuntimeError = error_type_t.RuntimeError


class ProblemCategory(IntEnum):
    LP = problem_category_t.LP
    MIP = problem_category_t.MIP
    IP = problem_category_t.IP


cdef char* c_get_string(string in_str):
    cdef char* c_string = <char *> malloc((in_str.length()+1) * sizeof(char))
    if not c_string:
        return NULL  # malloc failed
    # copy except the terminating char
    strcpy(c_string, in_str.c_str())
    return c_string


def get_data_ptr(array):
    if isinstance(array, cudf.Series):
        return array.__cuda_array_interface__['data'][0]
    elif isinstance(array, np.ndarray):
        return array.__array_interface__['data'][0]
    else:
        raise Exception(
            "get_data_ptr must be called with cudf.Series or np.ndarray"
        )


def type_cast(cudf_obj, np_type, name):
    if isinstance(cudf_obj, cudf.Series):
        cudf_type = cudf_obj.dtype
    elif isinstance(cudf_obj, np.ndarray):
        cudf_type = cudf_obj.dtype
    elif isinstance(cudf_obj, cudf.DataFrame):
        if all([np.issubdtype(dtype, np.number) for dtype in cudf_obj.dtypes]):  # noqa
            cudf_type = cudf_obj.dtypes[0]
        else:
            msg = "All columns in " + name + " should be numeric"
            raise Exception(msg)
    if ((np.issubdtype(np_type, np.floating) and
         (not np.issubdtype(cudf_type, np.floating)))
       or (np.issubdtype(np_type, np.integer) and
           (not np.issubdtype(cudf_type, np.integer)))
       or (np.issubdtype(np_type, np.bool_) and
           (not np.issubdtype(cudf_type, np.bool_)))
       or (np.issubdtype(np_type, np.int8) and
           (not np.issubdtype(cudf_type, np.int8)))):
        msg = "Casting " + name + " from " + str(cudf_type) + " to " + str(np.dtype(np_type))  # noqa
        warnings.warn(msg)
    cudf_obj = cudf_obj.astype(np.dtype(np_type))
    return cudf_obj


cdef set_data_model_view(DataModel data_model_obj):
    cdef data_model_view_t[int, double]* c_data_model_view = (
        data_model_obj.c_data_model_view.get()
    )

    # Set data_model_obj fields on the C++ side if set on the Python side
    cdef uintptr_t c_A_values = (
        get_data_ptr(data_model_obj.get_constraint_matrix_values())
    )
    cdef uintptr_t c_A_indices = (
        get_data_ptr(data_model_obj.get_constraint_matrix_indices())
    )
    cdef uintptr_t c_A_offsets = (
        get_data_ptr(data_model_obj.get_constraint_matrix_offsets())
    )
    if data_model_obj.get_constraint_matrix_values().shape[0] != 0 and data_model_obj.get_constraint_matrix_indices().shape[0] != 0 and data_model_obj.get_constraint_matrix_offsets().shape[0] != 0: # noqa
        c_data_model_view.set_csr_constraint_matrix(
            <const double *> c_A_values,
            data_model_obj.get_constraint_matrix_values().shape[0],
            <const int *> c_A_indices,
            data_model_obj.get_constraint_matrix_indices().shape[0],
            <const int *> c_A_offsets,
            data_model_obj.get_constraint_matrix_offsets().shape[0]
        )

    cdef uintptr_t c_b = (
        get_data_ptr(data_model_obj.get_constraint_bounds())
    )
    if data_model_obj.get_constraint_bounds().shape[0] != 0:
        c_data_model_view.set_constraint_bounds(
            <const double *> c_b,
            data_model_obj.get_constraint_bounds().shape[0]
        )

    cdef uintptr_t c_c = (
        get_data_ptr(data_model_obj.get_objective_coefficients())
    )
    if data_model_obj.get_objective_coefficients().shape[0] != 0:
        c_data_model_view.set_objective_coefficients(
            <const double *> c_c,
            data_model_obj.get_objective_coefficients().shape[0]
        )

    c_data_model_view.set_objective_scaling_factor(
        <double> data_model_obj.get_objective_scaling_factor()
    )
    c_data_model_view.set_objective_offset(
        <double> data_model_obj.get_objective_offset()
    )
    c_data_model_view.set_maximize(<bool> data_model_obj.maximize)

    cdef uintptr_t c_variable_lower_bounds = (
        get_data_ptr(data_model_obj.get_variable_lower_bounds())
    )
    if data_model_obj.get_variable_lower_bounds().shape[0] != 0:
        c_data_model_view.set_variable_lower_bounds(
            <const double *> c_variable_lower_bounds,
            data_model_obj.get_variable_lower_bounds().shape[0]
        )

    cdef uintptr_t c_variable_upper_bounds = (
        get_data_ptr(data_model_obj.get_variable_upper_bounds())
    )
    if data_model_obj.get_variable_upper_bounds().shape[0] != 0:
        c_data_model_view.set_variable_upper_bounds(
            <const double *> c_variable_upper_bounds,
            data_model_obj.get_variable_upper_bounds().shape[0]
        )
    cdef uintptr_t c_constraint_lower_bounds = (
        get_data_ptr(data_model_obj.get_constraint_lower_bounds())
    )
    if data_model_obj.get_constraint_lower_bounds().shape[0] != 0:
        c_data_model_view.set_constraint_lower_bounds(
            <const double *> c_constraint_lower_bounds,
            data_model_obj.get_constraint_lower_bounds().shape[0]
        )
    cdef uintptr_t c_constraint_upper_bounds = (
        get_data_ptr(data_model_obj.get_constraint_upper_bounds())
    )
    if data_model_obj.get_constraint_upper_bounds().shape[0] != 0:
        c_data_model_view.set_constraint_upper_bounds(
            <const double *> c_constraint_upper_bounds,
            data_model_obj.get_constraint_upper_bounds().shape[0]
        )
    cdef uintptr_t c_row_types = (
        get_data_ptr(data_model_obj.get_ascii_row_types())
    )
    if data_model_obj.get_ascii_row_types().shape[0] != 0:
        c_data_model_view.set_row_types(
            <const char *> c_row_types,
            data_model_obj.get_ascii_row_types().shape[0]
        )

    cdef uintptr_t c_var_types = (
        get_data_ptr(data_model_obj.get_variable_types())
    )
    if data_model_obj.get_variable_types().shape[0] != 0:
        c_data_model_view.set_variable_types(
            <const char *> c_var_types,
            data_model_obj.get_variable_types().shape[0]
        )

    # Set initial solution on the C++ side if set on the Python side
    cdef uintptr_t c_initial_primal_solution = (
        get_data_ptr(data_model_obj.get_initial_primal_solution())
    )
    if data_model_obj.get_initial_primal_solution().shape[0] != 0:
        c_data_model_view.set_initial_primal_solution(
            <const double *> c_initial_primal_solution,
            data_model_obj.get_initial_primal_solution().shape[0]
        )
    cdef uintptr_t c_initial_dual_solution = (
        get_data_ptr(data_model_obj.get_initial_dual_solution())
    )
    if data_model_obj.get_initial_dual_solution().shape[0] != 0:
        c_data_model_view.set_initial_dual_solution(
            <const double *> c_initial_dual_solution,
            data_model_obj.get_initial_dual_solution().shape[0]
        )


cdef set_solver_setting(
        unique_ptr[solver_settings_t[int, double]]& unique_solver_settings,
        settings,
        log_file,
        DataModel data_model_obj=None,
        mip=False):
    cdef solver_settings_t[int, double]* c_solver_settings = (
        unique_solver_settings.get()
    )
    # Set initial solution on the C++ side if set on the Python side
    cdef uintptr_t c_initial_primal_solution = (
        0 if data_model_obj is None else get_data_ptr(data_model_obj.get_initial_primal_solution())  # noqa
    )
    cdef uintptr_t c_initial_dual_solution = (
        0 if data_model_obj is None else get_data_ptr(data_model_obj.get_initial_dual_solution())  # noqa
    )

    cdef uintptr_t c_current_primal_solution
    cdef uintptr_t c_current_dual_solution
    cdef uintptr_t c_initial_primal_average
    cdef uintptr_t c_initial_dual_average
    cdef uintptr_t c_current_ATY
    cdef uintptr_t c_sum_primal_solutions
    cdef uintptr_t c_sum_dual_solutions
    cdef uintptr_t c_last_restart_duality_gap_primal_solution
    cdef uintptr_t c_last_restart_duality_gap_dual_solution
    cdef uintptr_t callback_ptr = 0
    if mip:
        if data_model_obj is not None and data_model_obj.get_initial_primal_solution().shape[0] != 0:  # noqa
            c_solver_settings.set_initial_mip_solution(
                <const double *> c_initial_primal_solution,
                data_model_obj.get_initial_primal_solution().shape[0]
            )

        for name, value in settings.settings_dict.items():
            c_solver_settings.set_parameter_from_string(
                name.encode('utf-8'),
                str(value).encode('utf-8')
            )

        callbacks = settings.get_mip_callbacks()
        for callback in callbacks:
            if callback:
                callback_ptr = callback.get_native_callback()

                c_solver_settings.set_mip_callback(
                    <base_solution_callback_t*>callback_ptr
                )
    else:
        if data_model_obj is not None and data_model_obj.get_initial_primal_solution().shape[0] != 0:  # noqa
            c_solver_settings.set_initial_pdlp_primal_solution(
                <const double *> c_initial_primal_solution,
                data_model_obj.get_initial_primal_solution().shape[0]
            )
        if data_model_obj is not None and data_model_obj.get_initial_dual_solution().shape[0] != 0: # noqa
            c_solver_settings.set_initial_pdlp_dual_solution(
                <const double *> c_initial_dual_solution,
                data_model_obj.get_initial_dual_solution().shape[0]
            )

        for name, value in settings.settings_dict.items():
            c_solver_settings.set_parameter_from_string(
                name.encode('utf-8'),
                str(value).encode('utf-8')
            )


    if settings.get_pdlp_warm_start_data() is not None:  # noqa
        if len(data_model_obj.get_objective_coefficients()) != len(
            settings.get_pdlp_warm_start_data().current_primal_solution
        ):
            raise Exception(
                "Invalid PDLPWarmStart data. Passed problem and PDLPWarmStart " # noqa
                "data should have the same amount of variables."
            )
        if len(data_model_obj.get_constraint_matrix_offsets()) - 1 != len( # noqa
            settings.get_pdlp_warm_start_data().current_dual_solution
        ):
            raise Exception(
                "Invalid PDLPWarmStart data. Passed problem and PDLPWarmStart " # noqa
                "data should have the same amount of constraints."
            )
        c_current_primal_solution = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().current_primal_solution # noqa
            )
        )
        c_current_dual_solution = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().current_dual_solution
            )
        )
        c_initial_primal_average = (
            get_data_ptr(
               settings.get_pdlp_warm_start_data().initial_primal_average # noqa
            )
        )
        c_initial_dual_average = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().initial_dual_average
            )
        )
        c_current_ATY = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().current_ATY
            )
        )
        c_sum_primal_solutions = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().sum_primal_solutions
            )
        )
        c_sum_dual_solutions = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().sum_dual_solutions
            )
        )
        c_last_restart_duality_gap_primal_solution = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution # noqa
            )
        )
        c_last_restart_duality_gap_dual_solution = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution # noqa
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
            settings.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution.shape[0], # Primal size # noqa
            settings.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution.shape[0], # Dual size # noqa
            settings.get_pdlp_warm_start_data().initial_primal_weight,
            settings.get_pdlp_warm_start_data().initial_step_size,
            settings.get_pdlp_warm_start_data().total_pdlp_iterations,
            settings.get_pdlp_warm_start_data().total_pdhg_iterations,
            settings.get_pdlp_warm_start_data().last_candidate_kkt_score,
            settings.get_pdlp_warm_start_data().last_restart_kkt_score,
            settings.get_pdlp_warm_start_data().sum_solution_weight,
            settings.get_pdlp_warm_start_data().iterations_since_last_restart # noqa
        )

    # Common to LP and MIP

    c_solver_settings.set_parameter_from_string(
        CUOPT_LOG_FILE.encode('utf-8'),
        log_file.encode('utf-8')
    )

cdef create_solution(unique_ptr[solver_ret_t] sol_ret_ptr,
                     DataModel data_model_obj,
                     is_batch=False):

    from cuopt.linear_programming.solution.solution import Solution

    sol_ret = move(sol_ret_ptr.get()[0])

    if sol_ret.problem_type == ProblemCategory.MIP or sol_ret.problem_type == ProblemCategory.IP: # noqa
        solution = DeviceBuffer.c_from_unique_ptr(
            move(sol_ret.mip_ret.solution_)
        )
        termination_status = sol_ret.mip_ret.termination_status_
        error_status = sol_ret.mip_ret.error_status_
        error_message = sol_ret.mip_ret.error_message_
        objective = sol_ret.mip_ret.objective_
        mip_gap = sol_ret.mip_ret.mip_gap_
        solution_bound = sol_ret.mip_ret.solution_bound_
        solve_time = sol_ret.mip_ret.total_solve_time_
        presolve_time = sol_ret.mip_ret.presolve_time_
        max_constraint_violation = sol_ret.mip_ret.max_constraint_violation_
        max_int_violation = sol_ret.mip_ret.max_int_violation_
        max_variable_bound_violation = sol_ret.mip_ret.max_variable_bound_violation_ # noqa
        num_nodes = sol_ret.mip_ret.nodes_
        num_simplex_iterations = sol_ret.mip_ret.simplex_iterations_

        solution = cudf.Series._from_column(
            cudf.core.column.build_column(
                as_buffer(solution),
                dtype=np.dtype(np.float64)
            )
        ).to_numpy()

        return Solution(
            ProblemCategory(sol_ret.problem_type),
            dict(zip(data_model_obj.get_variable_names(), solution)),
            solve_time,
            primal_solution=solution,
            termination_status=MILPTerminationStatus(termination_status),
            error_status=ErrorStatus(error_status),
            error_message=str(error_message),
            primal_objective=objective,
            mip_gap=mip_gap,
            solution_bound=solution_bound,
            presolve_time=presolve_time,
            max_variable_bound_violation=max_variable_bound_violation,
            max_int_violation=max_int_violation,
            max_constraint_violation=max_constraint_violation,
            num_nodes=num_nodes,
            num_simplex_iterations=num_simplex_iterations
        )

    else:
        primal_solution = DeviceBuffer.c_from_unique_ptr(
            move(sol_ret.lp_ret.primal_solution_)
        )
        dual_solution = DeviceBuffer.c_from_unique_ptr(move(sol_ret.lp_ret.dual_solution_)) # noqa
        reduced_cost = DeviceBuffer.c_from_unique_ptr(move(sol_ret.lp_ret.reduced_cost_)) # noqa

        primal_solution = cudf.Series._from_column(
            cudf.core.column.build_column(
                as_buffer(primal_solution),
                dtype=np.dtype(np.float64)
            )
        ).to_numpy()
        dual_solution = cudf.Series._from_column(
            cudf.core.column.build_column(
                as_buffer(dual_solution),
                dtype=np.dtype(np.float64)
            )
        ).to_numpy()
        reduced_cost = cudf.Series._from_column(
            cudf.core.column.build_column(
                as_buffer(reduced_cost),
                dtype=np.dtype(np.float64)
            )
        ).to_numpy()

        termination_status = sol_ret.lp_ret.termination_status_
        error_status = sol_ret.lp_ret.error_status_
        error_message = sol_ret.lp_ret.error_message_
        l2_primal_residual = sol_ret.lp_ret.l2_primal_residual_
        l2_dual_residual = sol_ret.lp_ret.l2_dual_residual_
        primal_objective = sol_ret.lp_ret.primal_objective_
        dual_objective = sol_ret.lp_ret.dual_objective_
        gap = sol_ret.lp_ret.gap_
        nb_iterations = sol_ret.lp_ret.nb_iterations_
        solve_time = sol_ret.lp_ret.solve_time_
        solved_by_pdlp = sol_ret.lp_ret.solved_by_pdlp_

        # In BatchSolve, we don't get the warm start data
        if not is_batch:
            current_primal_solution = DeviceBuffer.c_from_unique_ptr(
                move(sol_ret.lp_ret.current_primal_solution_)
            )
            current_dual_solution = DeviceBuffer.c_from_unique_ptr(
                move(sol_ret.lp_ret.current_dual_solution_)
            )
            initial_primal_average = DeviceBuffer.c_from_unique_ptr(
                move(sol_ret.lp_ret.initial_primal_average_)
            )
            initial_dual_average = DeviceBuffer.c_from_unique_ptr(
                move(sol_ret.lp_ret.initial_dual_average_)
            )
            current_ATY = DeviceBuffer.c_from_unique_ptr(
                move(sol_ret.lp_ret.current_ATY_)
            )
            sum_primal_solutions = DeviceBuffer.c_from_unique_ptr(
                move(sol_ret.lp_ret.sum_primal_solutions_)
            )
            sum_dual_solutions = DeviceBuffer.c_from_unique_ptr(
                move(sol_ret.lp_ret.sum_dual_solutions_)
            )
            last_restart_duality_gap_primal_solution = DeviceBuffer.c_from_unique_ptr( # noqa
                move(sol_ret.lp_ret.last_restart_duality_gap_primal_solution_)
            )
            last_restart_duality_gap_dual_solution = DeviceBuffer.c_from_unique_ptr( # noqa
                move(sol_ret.lp_ret.last_restart_duality_gap_dual_solution_)
            )
            initial_primal_weight = sol_ret.lp_ret.initial_primal_weight_
            initial_step_size = sol_ret.lp_ret.initial_step_size_
            total_pdlp_iterations = sol_ret.lp_ret.total_pdlp_iterations_
            total_pdhg_iterations = sol_ret.lp_ret.total_pdhg_iterations_
            last_candidate_kkt_score = sol_ret.lp_ret.last_candidate_kkt_score_
            last_restart_kkt_score = sol_ret.lp_ret.last_restart_kkt_score_
            sum_solution_weight = sol_ret.lp_ret.sum_solution_weight_
            iterations_since_last_restart = sol_ret.lp_ret.iterations_since_last_restart_ # noqa

            current_primal_solution = cudf.Series._from_column(
                cudf.core.column.build_column(
                    as_buffer(current_primal_solution),
                    dtype=np.dtype(np.float64)
                )
            ).to_numpy()
            current_dual_solution = cudf.Series._from_column(
                cudf.core.column.build_column(
                    as_buffer(current_dual_solution),
                    dtype=np.dtype(np.float64)
                )
            ).to_numpy()
            initial_primal_average = cudf.Series._from_column(
                cudf.core.column.build_column(
                    as_buffer(initial_primal_average),
                    dtype=np.dtype(np.float64)
                )
            ).to_numpy()
            initial_dual_average = cudf.Series._from_column(
                cudf.core.column.build_column(
                    as_buffer(initial_dual_average),
                    dtype=np.dtype(np.float64)
                )
            ).to_numpy()
            current_ATY = cudf.Series._from_column(
                cudf.core.column.build_column(
                    as_buffer(current_ATY),
                    dtype=np.dtype(np.float64)
                )
            ).to_numpy()
            sum_primal_solutions = cudf.Series._from_column(
                cudf.core.column.build_column(
                    as_buffer(sum_primal_solutions),
                    dtype=np.dtype(np.float64)
                )
            ).to_numpy()
            sum_dual_solutions = cudf.Series._from_column(
                cudf.core.column.build_column(
                    as_buffer(sum_dual_solutions),
                    dtype=np.dtype(np.float64)
                )
            ).to_numpy()
            last_restart_duality_gap_primal_solution = cudf.Series._from_column( # noqa
                cudf.core.column.build_column(
                    as_buffer(last_restart_duality_gap_primal_solution),
                    dtype=np.dtype(np.float64)
                )
            ).to_numpy()
            last_restart_duality_gap_dual_solution = cudf.Series._from_column(
                cudf.core.column.build_column(
                    as_buffer(last_restart_duality_gap_dual_solution),
                    dtype=np.dtype(np.float64)
                )
            ).to_numpy()

            return Solution(
                ProblemCategory(sol_ret.problem_type),
                dict(zip(data_model_obj.get_variable_names(), primal_solution)), # noqa
                solve_time,
                primal_solution,
                dual_solution,
                reduced_cost,
                current_primal_solution,
                current_dual_solution,
                initial_primal_average,
                initial_dual_average,
                current_ATY,
                sum_primal_solutions,
                sum_dual_solutions,
                last_restart_duality_gap_primal_solution,
                last_restart_duality_gap_dual_solution,
                initial_primal_weight,
                initial_step_size,
                total_pdlp_iterations,
                total_pdhg_iterations,
                last_candidate_kkt_score,
                last_restart_kkt_score,
                sum_solution_weight,
                iterations_since_last_restart,
                LPTerminationStatus(termination_status),
                ErrorStatus(error_status),
                str(error_message),
                l2_primal_residual,
                l2_dual_residual,
                primal_objective,
                dual_objective,
                gap,
                nb_iterations,
                solved_by_pdlp,
            )
        return Solution(
            problem_category=ProblemCategory(sol_ret.problem_type),
            vars=dict(zip(data_model_obj.get_variable_names(), primal_solution)), # noqa
            solve_time=solve_time,
            primal_solution=primal_solution,
            dual_solution=dual_solution,
            reduced_cost=reduced_cost,
            termination_status=LPTerminationStatus(termination_status),
            error_status=ErrorStatus(error_status),
            error_message=str(error_message),
            primal_residual=l2_primal_residual,
            dual_residual=l2_dual_residual,
            primal_objective=primal_objective,
            dual_objective=dual_objective,
            gap=gap,
            nb_iterations=nb_iterations,
            solved_by_pdlp=solved_by_pdlp,
        )


def Solve(py_data_model_obj, settings, str log_file, mip=False):

    cdef DataModel data_model_obj = <DataModel>py_data_model_obj
    cdef unique_ptr[solver_settings_t[int, double]] unique_solver_settings

    unique_solver_settings.reset(new solver_settings_t[int, double]())

    data_model_obj.variable_types = type_cast(
        data_model_obj.variable_types, "S1", "variable_types"
    )

    set_solver_setting(
        unique_solver_settings, settings, log_file, data_model_obj, mip
    )
    set_data_model_view(data_model_obj)

    return create_solution(move(call_solve(
        data_model_obj.c_data_model_view.get(),
        unique_solver_settings.get(),
    )), data_model_obj)


cdef insert_vector(DataModel data_model_obj,
                   vector[data_model_view_t[int, double] *]& data_model_views):
    data_model_views.push_back(data_model_obj.c_data_model_view.get())


def BatchSolve(py_data_model_list, settings, str log_file):
    cdef unique_ptr[solver_settings_t[int, double]] unique_solver_settings
    unique_solver_settings.reset(new solver_settings_t[int, double]())

    if settings.get_pdlp_warm_start_data() is not None:  # noqa
        raise Exception("Cannot use warmstart data with Batch Solve")
    set_solver_setting(unique_solver_settings, settings, log_file)

    cdef vector[data_model_view_t[int, double] *] data_model_views

    for data_model_obj in py_data_model_list:
        set_data_model_view(<DataModel>data_model_obj)
        insert_vector(<DataModel>data_model_obj, data_model_views)

    cdef pair[
        vector[unique_ptr[solver_ret_t]],
        double] batch_solve_result = (
        move(call_batch_solve(data_model_views, unique_solver_settings.get())) # noqa
    )

    cdef vector[unique_ptr[solver_ret_t]] c_solutions = (
        move(batch_solve_result.first)
    )
    cdef double solve_time = batch_solve_result.second

    solutions = [] * len(py_data_model_list)
    for i in range(c_solutions.size()):
        solutions.append(
            create_solution(
                move(c_solutions[i]),
                <DataModel>py_data_model_list[i],
                True
            )
        )

    return solutions, solve_time
