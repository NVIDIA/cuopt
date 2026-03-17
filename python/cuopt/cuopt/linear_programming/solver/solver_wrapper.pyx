# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0


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
    get_cpu_lp_solutions,
    get_cpu_mip_solution,
    get_gpu_lp_solutions,
    get_gpu_mip_solution,
    linear_programming_ret_t,
    lp_cpu_solutions_t,
    lp_gpu_solutions_t,
    mip_ret_t,
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

from cuopt.linear_programming.solver_settings.solver_settings import (
    PDLPSolverMode,
    SolverSettings,
)
from cuopt.utilities import InputValidationError, series_from_buf

import pyarrow as pa


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


cdef object _vector_to_numpy(const vector[double]& vec):
    """Convert C++ std::vector<double> to numpy array"""
    cdef Py_ssize_t size = vec.size()
    if size == 0:
        return np.array([], dtype=np.float64)
    cdef const double* data_ptr = vec.data()
    return np.asarray(<double[:size]> data_ptr, dtype=np.float64).copy()


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


cdef set_solver_setting(
        unique_ptr[solver_settings_t[int, double]]& unique_solver_settings,
        settings,
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
    cdef uintptr_t callback_user_data = 0
    if mip:
        if data_model_obj is not None and data_model_obj.get_initial_primal_solution().shape[0] != 0:  # noqa
            c_solver_settings.add_initial_mip_solution(
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
                callback_user_data = (
                    callback.get_user_data_ptr()
                    if hasattr(callback, "get_user_data_ptr")
                    else 0
                )

                c_solver_settings.set_mip_callback(
                    <base_solution_callback_t*>callback_ptr,
                    <void*>callback_user_data
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
        warm_start_data = settings.get_pdlp_warm_start_data()
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

cdef create_solution(unique_ptr[solver_ret_t] sol_ret_ptr,
                     DataModel data_model_obj,
                     is_batch=False):

    from cuopt.linear_programming.solution.solution import Solution

    cdef solver_ret_t* sol_ret = sol_ret_ptr.get()

    cdef mip_ret_t* mip_ptr
    cdef linear_programming_ret_t* lp_ptr
    cdef lp_gpu_solutions_t* gpu_sols
    cdef lp_cpu_solutions_t* cpu_sols

    if sol_ret.problem_type == ProblemCategory.MIP or sol_ret.problem_type == ProblemCategory.IP: # noqa
        mip_ptr = &sol_ret.mip_ret

        # Extract solution vector — branch only for the buffer type
        if mip_ptr.is_gpu():
            solution_buf = DeviceBuffer.c_from_unique_ptr(move(get_gpu_mip_solution(mip_ptr[0]))) # noqa
            solution = series_from_buf(solution_buf, pa.float64()).to_numpy()
        else:
            solution = _vector_to_numpy(get_cpu_mip_solution(mip_ptr[0]))

        return Solution(
            ProblemCategory(sol_ret.problem_type),
            dict(zip(data_model_obj.get_variable_names(), solution)),
            mip_ptr.total_solve_time_,
            primal_solution=solution,
            termination_status=MILPTerminationStatus(mip_ptr.termination_status_),
            error_status=ErrorStatus(mip_ptr.error_status_),
            error_message=mip_ptr.error_message_.decode('utf-8'),
            primal_objective=mip_ptr.objective_,
            mip_gap=mip_ptr.mip_gap_,
            solution_bound=mip_ptr.solution_bound_,
            presolve_time=mip_ptr.presolve_time_,
            max_variable_bound_violation=mip_ptr.max_variable_bound_violation_,
            max_int_violation=mip_ptr.max_int_violation_,
            max_constraint_violation=mip_ptr.max_constraint_violation_,
            num_nodes=mip_ptr.nodes_,
            num_simplex_iterations=mip_ptr.simplex_iterations_
        )

    else:
        lp_ptr = &sol_ret.lp_ret

        # Extract solution vectors — branch only for the buffer type
        if lp_ptr.is_gpu():
            gpu_sols = &get_gpu_lp_solutions(lp_ptr[0])

            primal_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.primal_solution_)) # noqa
            primal_solution = series_from_buf(primal_buf, pa.float64()).to_numpy() # noqa

            dual_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.dual_solution_)) # noqa
            dual_solution = series_from_buf(dual_buf, pa.float64()).to_numpy()

            reduced_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.reduced_cost_)) # noqa
            reduced_cost = series_from_buf(reduced_buf, pa.float64()).to_numpy()

            if not is_batch:
                if gpu_sols.current_primal_solution_.get() != NULL and gpu_sols.current_primal_solution_.get()[0].size() > 0: # noqa
                    current_primal_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.current_primal_solution_)) # noqa
                    current_primal = series_from_buf(current_primal_buf, pa.float64()).to_numpy() # noqa

                    current_dual_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.current_dual_solution_)) # noqa
                    current_dual = series_from_buf(current_dual_buf, pa.float64()).to_numpy() # noqa

                    initial_primal_avg_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.initial_primal_average_)) # noqa
                    initial_primal_avg = series_from_buf(initial_primal_avg_buf, pa.float64()).to_numpy() # noqa

                    initial_dual_avg_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.initial_dual_average_)) # noqa
                    initial_dual_avg = series_from_buf(initial_dual_avg_buf, pa.float64()).to_numpy() # noqa

                    current_ATY_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.current_ATY_)) # noqa
                    current_ATY = series_from_buf(current_ATY_buf, pa.float64()).to_numpy() # noqa

                    sum_primal_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.sum_primal_solutions_)) # noqa
                    sum_primal = series_from_buf(sum_primal_buf, pa.float64()).to_numpy() # noqa

                    sum_dual_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.sum_dual_solutions_)) # noqa
                    sum_dual = series_from_buf(sum_dual_buf, pa.float64()).to_numpy() # noqa

                    last_restart_primal_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.last_restart_duality_gap_primal_solution_)) # noqa
                    last_restart_primal = series_from_buf(last_restart_primal_buf, pa.float64()).to_numpy() # noqa

                    last_restart_dual_buf = DeviceBuffer.c_from_unique_ptr(move(gpu_sols.last_restart_duality_gap_dual_solution_)) # noqa
                    last_restart_dual = series_from_buf(last_restart_dual_buf, pa.float64()).to_numpy() # noqa
                else:
                    current_primal = None
                    current_dual = None
                    initial_primal_avg = None
                    initial_dual_avg = None
                    current_ATY = None
                    sum_primal = None
                    sum_dual = None
                    last_restart_primal = None
                    last_restart_dual = None

        else:
            cpu_sols = &get_cpu_lp_solutions(lp_ptr[0])

            primal_solution = _vector_to_numpy(cpu_sols.primal_solution_)
            dual_solution = _vector_to_numpy(cpu_sols.dual_solution_)
            reduced_cost = _vector_to_numpy(cpu_sols.reduced_cost_)

            if not is_batch:
                if cpu_sols.current_primal_solution_.size() > 0:
                    current_primal = _vector_to_numpy(cpu_sols.current_primal_solution_) # noqa
                    current_dual = _vector_to_numpy(cpu_sols.current_dual_solution_) # noqa
                    initial_primal_avg = _vector_to_numpy(cpu_sols.initial_primal_average_) # noqa
                    initial_dual_avg = _vector_to_numpy(cpu_sols.initial_dual_average_) # noqa
                    current_ATY = _vector_to_numpy(cpu_sols.current_ATY_) # noqa
                    sum_primal = _vector_to_numpy(cpu_sols.sum_primal_solutions_) # noqa
                    sum_dual = _vector_to_numpy(cpu_sols.sum_dual_solutions_) # noqa
                    last_restart_primal = _vector_to_numpy(cpu_sols.last_restart_duality_gap_primal_solution_) # noqa
                    last_restart_dual = _vector_to_numpy(cpu_sols.last_restart_duality_gap_dual_solution_) # noqa
                else:
                    current_primal = None
                    current_dual = None
                    initial_primal_avg = None
                    initial_dual_avg = None
                    current_ATY = None
                    sum_primal = None
                    sum_dual = None
                    last_restart_primal = None
                    last_restart_dual = None

        # Shared scalar access — written once regardless of GPU/CPU backend
        if not is_batch:
            return Solution(
                ProblemCategory(sol_ret.problem_type),
                dict(zip(data_model_obj.get_variable_names(), primal_solution)),
                lp_ptr.solve_time_,
                primal_solution,
                dual_solution,
                reduced_cost,
                current_primal,
                current_dual,
                initial_primal_avg,
                initial_dual_avg,
                current_ATY,
                sum_primal,
                sum_dual,
                last_restart_primal,
                last_restart_dual,
                lp_ptr.initial_primal_weight_,
                lp_ptr.initial_step_size_,
                lp_ptr.total_pdlp_iterations_,
                lp_ptr.total_pdhg_iterations_,
                lp_ptr.last_candidate_kkt_score_,
                lp_ptr.last_restart_kkt_score_,
                lp_ptr.sum_solution_weight_,
                lp_ptr.iterations_since_last_restart_,
                LPTerminationStatus(lp_ptr.termination_status_),
                ErrorStatus(lp_ptr.error_status_),
                lp_ptr.error_message_.decode('utf-8'),
                lp_ptr.l2_primal_residual_,
                lp_ptr.l2_dual_residual_,
                lp_ptr.primal_objective_,
                lp_ptr.dual_objective_,
                lp_ptr.gap_,
                lp_ptr.nb_iterations_,
                lp_ptr.solved_by_pdlp_,
            )
        else:
            return Solution(
                problem_category=ProblemCategory(sol_ret.problem_type),
                vars=dict(zip(data_model_obj.get_variable_names(), primal_solution)),
                solve_time=lp_ptr.solve_time_,
                primal_solution=primal_solution,
                dual_solution=dual_solution,
                reduced_cost=reduced_cost,
                termination_status=LPTerminationStatus(lp_ptr.termination_status_),
                error_status=ErrorStatus(lp_ptr.error_status_),
                error_message=lp_ptr.error_message_.decode('utf-8'),
                primal_residual=lp_ptr.l2_primal_residual_,
                dual_residual=lp_ptr.l2_dual_residual_,
                primal_objective=lp_ptr.primal_objective_,
                dual_objective=lp_ptr.dual_objective_,
                gap=lp_ptr.gap_,
                nb_iterations=lp_ptr.nb_iterations_,
                solved_by_pdlp=lp_ptr.solved_by_pdlp_,
            )


def Solve(py_data_model_obj, settings, mip=False):

    cdef DataModel data_model_obj = <DataModel>py_data_model_obj
    cdef unique_ptr[solver_settings_t[int, double]] unique_solver_settings

    unique_solver_settings.reset(new solver_settings_t[int, double]())

    data_model_obj.variable_types = type_cast(
        data_model_obj.variable_types, "S1", "variable_types"
    )

    set_solver_setting(
        unique_solver_settings, settings, data_model_obj, mip
    )
    data_model_obj.set_data_model_view()

    return create_solution(move(call_solve(
        data_model_obj.c_data_model_view.get(),
        unique_solver_settings.get(),
    )), data_model_obj)


cdef set_and_insert_vector(
        DataModel data_model_obj,
        vector[data_model_view_t[int, double] *]& data_model_views):
    data_model_obj.set_data_model_view()
    data_model_views.push_back(data_model_obj.c_data_model_view.get())


def BatchSolve(py_data_model_list, settings):
    cdef unique_ptr[solver_settings_t[int, double]] unique_solver_settings
    unique_solver_settings.reset(new solver_settings_t[int, double]())

    if settings.get_pdlp_warm_start_data() is not None:  # noqa
        raise Exception("Cannot use warmstart data with Batch Solve")
    set_solver_setting(unique_solver_settings, settings)

    cdef vector[data_model_view_t[int, double] *] data_model_views

    for data_model_obj in py_data_model_list:
        set_and_insert_vector(<DataModel>data_model_obj, data_model_views)

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
