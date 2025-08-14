# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

import logging
import os
import time

from fastapi import HTTPException

from cuopt import linear_programming
from cuopt.linear_programming.internals import GetSolutionCallback
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_ABSOLUTE_DUAL_TOLERANCE,
    CUOPT_ABSOLUTE_GAP_TOLERANCE,
    CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
    CUOPT_CROSSOVER,
    CUOPT_DUAL_INFEASIBLE_TOLERANCE,
    CUOPT_FIRST_PRIMAL_FEASIBLE,
    CUOPT_INFEASIBILITY_DETECTION,
    CUOPT_ITERATION_LIMIT,
    CUOPT_LOG_FILE,
    CUOPT_LOG_TO_CONSOLE,
    CUOPT_METHOD,
    CUOPT_MIP_ABSOLUTE_GAP,
    CUOPT_MIP_ABSOLUTE_TOLERANCE,
    CUOPT_MIP_HEURISTICS_ONLY,
    CUOPT_MIP_INTEGRALITY_TOLERANCE,
    CUOPT_MIP_RELATIVE_GAP,
    CUOPT_MIP_RELATIVE_TOLERANCE,
    CUOPT_MIP_SCALING,
    CUOPT_NUM_CPU_THREADS,
    CUOPT_PDLP_SOLVER_MODE,
    CUOPT_PER_CONSTRAINT_RESIDUAL,
    CUOPT_PRESOLVE,
    CUOPT_PRIMAL_INFEASIBLE_TOLERANCE,
    CUOPT_RELATIVE_DUAL_TOLERANCE,
    CUOPT_RELATIVE_GAP_TOLERANCE,
    CUOPT_RELATIVE_PRIMAL_TOLERANCE,
    CUOPT_SAVE_BEST_PRIMAL_SO_FAR,
    CUOPT_STRICT_INFEASIBILITY,
    CUOPT_TIME_LIMIT,
)
from cuopt.linear_programming.solver.solver_wrapper import (
    ErrorStatus,
    LPTerminationStatus,
    MILPTerminationStatus,
)
from cuopt.utilities import (
    InputRuntimeError,
    InputValidationError,
    OutOfMemoryError,
)


def dep_warning(field):
    return (
        f"solver config {field} is deprecated and will "
        "be removed in a future release"
    )


def ignored_warning(field):
    return f"solver config {field} ignored in the cuopt service"


class CustomGetSolutionCallback(GetSolutionCallback):
    def __init__(self, sender, req_id):
        super().__init__()
        self.req_id = req_id
        self.sender = sender

    def get_solution(self, solution, solution_cost):
        self.sender(
            self.req_id,
            solution.copy_to_host(),
            solution_cost.copy_to_host()[0],
        )


def warn_on_objectives(solver_config):
    warnings = []
    return warnings, solver_config


def create_data_model(LP_data):

    warnings = []

    # Create data model object
    data_model = linear_programming.DataModel()

    csr_constraint_matrix = LP_data.csr_constraint_matrix
    data_model.set_csr_constraint_matrix(
        csr_constraint_matrix.values,
        csr_constraint_matrix.indices,
        csr_constraint_matrix.offsets,
    )

    constraint_bounds = LP_data.constraint_bounds
    if constraint_bounds.bounds is not None:
        data_model.set_constraint_bounds(constraint_bounds.bounds)
    if constraint_bounds.types is not None:
        if len(constraint_bounds.types):
            data_model.set_row_types(constraint_bounds.types)
    if constraint_bounds.upper_bounds is not None:
        if len(constraint_bounds.upper_bounds):
            data_model.set_constraint_upper_bounds(
                constraint_bounds.upper_bounds
            )
    if constraint_bounds.lower_bounds is not None:
        if len(constraint_bounds.lower_bounds):
            data_model.set_constraint_lower_bounds(
                constraint_bounds.lower_bounds
            )

    objective_data = LP_data.objective_data
    if objective_data.coefficients is not None:
        data_model.set_objective_coefficients(objective_data.coefficients)
    if objective_data.scalability_factor is not None:
        data_model.set_objective_scaling_factor(
            objective_data.scalability_factor
        )
    if objective_data.offset is not None:
        data_model.set_objective_offset(objective_data.offset)

    variable_bounds = LP_data.variable_bounds
    if variable_bounds.upper_bounds is not None:
        data_model.set_variable_upper_bounds(variable_bounds.upper_bounds)
    if variable_bounds.lower_bounds is not None:
        data_model.set_variable_lower_bounds(variable_bounds.lower_bounds)

    initial_sol = LP_data.initial_solution
    if initial_sol is not None:
        if initial_sol.primal is not None:
            data_model.set_initial_primal_solution(initial_sol.primal)
        if initial_sol.dual is not None:
            data_model.set_initial_dual_solution(initial_sol.dual)

    if LP_data.maximize is not None:
        data_model.set_maximize(LP_data.maximize)

    if LP_data.variable_types is not None:
        data_model.set_variable_types(LP_data.variable_types)

    if LP_data.variable_names is not None:
        data_model.set_variable_names(LP_data.variable_names)

    return warnings, data_model


def create_solver(LP_data, warmstart_data):
    warnings = []
    solver_settings = linear_programming.SolverSettings()

    if LP_data.solver_config is not None:
        solver_config = LP_data.solver_config
        if solver_config.infeasibility_detection is not None:
            solver_settings.set_parameter(
                CUOPT_INFEASIBILITY_DETECTION,
                solver_config.infeasibility_detection,
            )
        if solver_config.solver_mode is not None:
            solver_settings.set_parameter(
                CUOPT_PDLP_SOLVER_MODE,
                linear_programming.solver_settings.PDLPSolverMode(
                    solver_config.solver_mode
                ),
            )
            warnings.append(dep_warning("solver_mode"))
        elif solver_config.pdlp_solver_mode is not None:
            solver_settings.set_parameter(
                CUOPT_PDLP_SOLVER_MODE,
                linear_programming.solver_settings.PDLPSolverMode(
                    solver_config.pdlp_solver_mode
                ),
            )
        if solver_config.method is not None:
            solver_settings.set_parameter(
                CUOPT_METHOD,
                linear_programming.solver_settings.SolverMethod(
                    solver_config.method
                ),
            )
        if solver_config.crossover is not None:
            solver_settings.set_parameter(
                CUOPT_CROSSOVER, solver_config.crossover
            )
        try:
            lp_time_limit = float(os.environ.get("CUOPT_LP_TIME_LIMIT_SEC"))
        except Exception:
            lp_time_limit = None
        if solver_config.time_limit is None:
            time_limit = lp_time_limit
        elif lp_time_limit:
            time_limit = min(solver_config.time_limit, lp_time_limit)
        else:
            time_limit = solver_config.time_limit
        if time_limit is not None:
            logging.debug(f"setting LP time limit to {time_limit}sec")
            solver_settings.set_parameter(CUOPT_TIME_LIMIT, time_limit)

        try:
            lp_iteration_limit = int(
                os.environ.get("CUOPT_LP_ITERATION_LIMIT")
            )
        except Exception:
            lp_iteration_limit = None
        if solver_config.iteration_limit is None:
            iteration_limit = lp_iteration_limit
        elif lp_iteration_limit:
            iteration_limit = min(
                solver_config.iteration_limit, lp_iteration_limit
            )
        else:
            iteration_limit = solver_config.iteration_limit
        if iteration_limit is not None:
            logging.debug(f"setting LP iteration limit to {iteration_limit}")
            solver_settings.set_parameter(
                CUOPT_ITERATION_LIMIT, iteration_limit
            )

        if solver_config.tolerances is not None:
            tolerance = solver_config.tolerances
            if tolerance.optimality is not None:
                solver_settings.set_optimality_tolerance(tolerance.optimality)
            if tolerance.absolute_dual_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_ABSOLUTE_DUAL_TOLERANCE,
                    tolerance.absolute_dual_tolerance,
                )
            elif tolerance.absolute_dual is not None:
                solver_settings.set_parameter(
                    CUOPT_ABSOLUTE_DUAL_TOLERANCE, tolerance.absolute_dual
                )
                warnings.append(dep_warning("absolute_dual"))
            if tolerance.absolute_primal_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
                    tolerance.absolute_primal_tolerance,
                )
            elif tolerance.absolute_primal is not None:
                solver_settings.set_parameter(
                    CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, tolerance.absolute_primal
                )
                warnings.append(dep_warning("absolute_primal"))
            if tolerance.absolute_gap_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_ABSOLUTE_GAP_TOLERANCE,
                    tolerance.absolute_gap_tolerance,
                )
            elif tolerance.absolute_gap is not None:
                solver_settings.set_parameter(
                    CUOPT_ABSOLUTE_GAP_TOLERANCE, tolerance.absolute_gap
                )
                warnings.append(dep_warning("absolute_gap"))
            if tolerance.relative_dual_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_RELATIVE_DUAL_TOLERANCE,
                    tolerance.relative_dual_tolerance,
                )
            elif tolerance.relative_dual is not None:
                solver_settings.set_parameter(
                    CUOPT_RELATIVE_DUAL_TOLERANCE, tolerance.relative_dual
                )
                warnings.append(dep_warning("relative_dual"))
            if tolerance.relative_primal_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_RELATIVE_PRIMAL_TOLERANCE,
                    tolerance.relative_primal_tolerance,
                )
            elif tolerance.relative_primal is not None:
                solver_settings.set_parameter(
                    CUOPT_RELATIVE_PRIMAL_TOLERANCE, tolerance.relative_primal
                )
                warnings.append(dep_warning("relative_primal"))
            if tolerance.relative_gap_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_RELATIVE_GAP_TOLERANCE,
                    tolerance.relative_gap_tolerance,
                )
            elif tolerance.relative_gap is not None:
                solver_settings.set_parameter(
                    CUOPT_RELATIVE_GAP_TOLERANCE, tolerance.relative_gap
                )
                warnings.append(dep_warning("relative_gap"))
            if tolerance.primal_infeasible_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_PRIMAL_INFEASIBLE_TOLERANCE,
                    tolerance.primal_infeasible_tolerance,
                )
            elif tolerance.primal_infeasible is not None:
                solver_settings.set_parameter(
                    CUOPT_PRIMAL_INFEASIBLE_TOLERANCE,
                    tolerance.primal_infeasible,
                )
                warnings.append(dep_warning("primal_infeasible"))
            if tolerance.dual_infeasible_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_DUAL_INFEASIBLE_TOLERANCE,
                    tolerance.dual_infeasible_tolerance,
                )
            elif tolerance.dual_infeasible is not None:
                solver_settings.set_parameter(
                    CUOPT_DUAL_INFEASIBLE_TOLERANCE, tolerance.dual_infeasible
                )
                warnings.append(dep_warning("dual_infeasible"))
            if tolerance.mip_integrality_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_MIP_INTEGRALITY_TOLERANCE,
                    tolerance.mip_integrality_tolerance,
                )
            elif tolerance.integrality_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_MIP_INTEGRALITY_TOLERANCE,
                    tolerance.integrality_tolerance,
                )
                warnings.append(dep_warning("integrality_tolerance"))
            if tolerance.mip_absolute_gap is not None:
                solver_settings.set_parameter(
                    CUOPT_MIP_ABSOLUTE_GAP, tolerance.mip_absolute_gap
                )
            elif tolerance.absolute_mip_gap is not None:
                solver_settings.set_parameter(
                    CUOPT_MIP_ABSOLUTE_GAP, tolerance.absolute_mip_gap
                )
                warnings.append(dep_warning("absolute_mip_gap"))
            if tolerance.mip_relative_gap is not None:
                solver_settings.set_parameter(
                    CUOPT_MIP_RELATIVE_GAP, tolerance.mip_relative_gap
                )
            elif tolerance.relative_mip_gap is not None:
                solver_settings.set_parameter(
                    CUOPT_MIP_RELATIVE_GAP, tolerance.relative_mip_gap
                )
                warnings.append(dep_warning("relative_mip_gap"))
            if tolerance.mip_absolute_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_MIP_ABSOLUTE_TOLERANCE,
                    tolerance.mip_absolute_tolerance,
                )
            if tolerance.mip_relative_tolerance is not None:
                solver_settings.set_parameter(
                    CUOPT_MIP_RELATIVE_TOLERANCE,
                    tolerance.mip_relative_tolerance,
                )
        if warmstart_data is not None:
            solver_settings.set_pdlp_warm_start_data(warmstart_data)
        if solver_config.mip_scaling is not None:
            solver_settings.set_parameter(
                CUOPT_MIP_SCALING, solver_config.mip_scaling
            )
        if solver_config.heuristics_only is not None:
            solver_settings.set_parameter(
                CUOPT_MIP_HEURISTICS_ONLY, solver_config.heuristics_only
            )
            warnings.append(dep_warning("heuristics_only"))
        elif solver_config.mip_heuristics_only is not None:
            solver_settings.set_parameter(
                CUOPT_MIP_HEURISTICS_ONLY, solver_config.mip_heuristics_only
            )
        if solver_config.num_cpu_threads is not None:
            solver_settings.set_parameter(
                CUOPT_NUM_CPU_THREADS, solver_config.num_cpu_threads
            )
        if solver_config.crossover is not None:
            solver_settings.set_parameter(
                CUOPT_CROSSOVER, solver_config.crossover
            )

        def is_mip(var_types):
            if var_types is None or len(var_types) == 0:
                return False
            elif "I" in var_types:
                return True

            return False

        if solver_config.presolve is None:
            if is_mip(LP_data.variable_types):
                solver_config.presolve = True
            else:
                solver_config.presolve = False

        if solver_config.presolve is not None:
            solver_settings.set_parameter(
                CUOPT_PRESOLVE, solver_config.presolve
            )

        if solver_config.log_to_console is not None:
            solver_settings.set_parameter(
                CUOPT_LOG_TO_CONSOLE, solver_config.log_to_console
            )
        if solver_config.strict_infeasibility is not None:
            solver_settings.set_parameter(
                CUOPT_STRICT_INFEASIBILITY, solver_config.strict_infeasibility
            )
        if solver_config.user_problem_file != "":
            warnings.append(ignored_warning("user_problem_file"))
        if solver_config.per_constraint_residual is not None:
            solver_settings.set_parameter(
                CUOPT_PER_CONSTRAINT_RESIDUAL,
                solver_config.per_constraint_residual,
            )
        if solver_config.save_best_primal_so_far is not None:
            solver_settings.set_parameter(
                CUOPT_SAVE_BEST_PRIMAL_SO_FAR,
                solver_config.save_best_primal_so_far,
            )
        if solver_config.first_primal_feasible is not None:
            solver_settings.set_parameter(
                CUOPT_FIRST_PRIMAL_FEASIBLE,
                solver_config.first_primal_feasible,
            )
        if solver_config.log_file != "":
            solver_settings.set_parameter(
                CUOPT_LOG_FILE, solver_config.log_file
            )
        if solver_config.solution_file != "":
            warnings.append(ignored_warning("solution_file"))

    return warnings, solver_settings


def get_solver_exception_type(status, message):
    msg = f"error_status: {status}, msg: {message}"

    # TODO change these to enums once we have a clear place
    # to map them from for both routing and lp
    if status == ErrorStatus.Success:
        return None
    elif status == ErrorStatus.ValidationError:
        return InputValidationError(msg)
    elif status == ErrorStatus.OutOfMemoryError:
        return OutOfMemoryError(msg)
    elif status == ErrorStatus.RuntimeError:
        return InputRuntimeError(msg)
    else:
        return RuntimeError(msg)


def solve(LP_data, reqId, intermediate_sender, warmstart_data):
    notes = []

    def get_if_attribute_is_valid_else_none(attr):
        try:
            return attr()
        except AttributeError:
            return None

    def extract_pdlpwarmstart_data(data):
        if data is None:
            return None
        pdlpwarmstart_data = {
            "current_primal_solution": data.current_primal_solution,
            "current_dual_solution": data.current_dual_solution,
            "initial_primal_average": data.initial_primal_average,
            "initial_dual_average": data.initial_dual_average,
            "current_ATY": data.current_ATY,
            "sum_primal_solutions": data.sum_primal_solutions,
            "sum_dual_solutions": data.sum_dual_solutions,
            "last_restart_duality_gap_primal_solution": data.last_restart_duality_gap_primal_solution,  # noqa
            "last_restart_duality_gap_dual_solution": data.last_restart_duality_gap_dual_solution,  # noqa
            "initial_primal_weight": data.initial_primal_weight,
            "initial_step_size": data.initial_step_size,
            "total_pdlp_iterations": data.total_pdlp_iterations,
            "total_pdhg_iterations": data.total_pdhg_iterations,
            "last_candidate_kkt_score": data.last_candidate_kkt_score,
            "last_restart_kkt_score": data.last_restart_kkt_score,
            "sum_solution_weight": data.sum_solution_weight,
            "iterations_since_last_restart": data.iterations_since_last_restart,  # noqa
        }
        return pdlpwarmstart_data

    def create_solution(sol):
        solution = {}
        status = sol.get_termination_status()
        if status in (
            LPTerminationStatus.Optimal,
            LPTerminationStatus.IterationLimit,
            LPTerminationStatus.TimeLimit,
            MILPTerminationStatus.Optimal,
            MILPTerminationStatus.FeasibleFound,
        ):

            primal_solution = get_if_attribute_is_valid_else_none(
                sol.get_primal_solution
            )
            primal_solution = (
                primal_solution
                if primal_solution is None
                else primal_solution.tolist()
            )
            dual_solution = get_if_attribute_is_valid_else_none(
                sol.get_dual_solution
            )
            dual_solution = (
                dual_solution
                if dual_solution is None
                else dual_solution.tolist()
            )
            lp_stats = get_if_attribute_is_valid_else_none(sol.get_lp_stats)
            reduced_cost = get_if_attribute_is_valid_else_none(
                sol.get_reduced_cost
            )
            reduced_cost = (
                reduced_cost if reduced_cost is None else reduced_cost.tolist()
            )
            milp_stats = get_if_attribute_is_valid_else_none(
                sol.get_milp_stats
            )
            solution["problem_category"] = sol.get_problem_category().name
            solution["primal_solution"] = primal_solution
            solution["dual_solution"] = dual_solution
            solution["primal_objective"] = get_if_attribute_is_valid_else_none(
                sol.get_primal_objective
            )
            solution["dual_objective"] = get_if_attribute_is_valid_else_none(
                sol.get_dual_objective
            )
            solution["solver_time"] = sol.get_solve_time()
            solution["solved_by_pdlp"] = sol.get_solved_by_pdlp()
            solution["vars"] = sol.get_vars()
            solution["lp_statistics"] = {} if lp_stats is None else lp_stats
            solution["reduced_cost"] = reduced_cost
            solution["pdlpwarmstart_data"] = extract_pdlpwarmstart_data(
                sol.get_pdlp_warm_start_data()
            )
            solution["milp_statistics"] = (
                {} if milp_stats is None else milp_stats
            )

        res = {
            "status": status.name,
            "solution": solution,
        }
        notes.append(sol.get_termination_reason())
        return res

    try:
        is_batch = False
        sol = None
        total_solve_time = None
        if type(LP_data) is list:
            is_batch = True
            data_model_list = []
            warnings = []
            for i_data in LP_data:
                i_warnings, data_model = create_data_model(i_data)
                data_model_list.append(data_model)
                warnings.extend(i_warnings)
            cswarnings, solver_settings = create_solver(
                LP_data[0], warmstart_data
            )
            warnings.extend(cswarnings)
            sol, total_solve_time = linear_programming.BatchSolve(
                data_model_list, solver_settings
            )
        else:
            warnings, data_model = create_data_model(LP_data)
            cswarnings, solver_settings = create_solver(
                LP_data, warmstart_data
            )
            warnings.extend(cswarnings)
            callback = (
                CustomGetSolutionCallback(intermediate_sender, reqId)
                if intermediate_sender is not None
                else None
            )
            solver_settings.set_mip_callback(callback)
            solve_begin_time = time.time()
            sol = linear_programming.Solve(
                data_model, solver_settings=solver_settings
            )
            total_solve_time = time.time() - solve_begin_time

        res = None
        if is_batch:
            res = []
            for i_sol in sol:
                if i_sol is None:
                    continue
                if i_sol.get_error_status() != ErrorStatus.Success:
                    res.append(
                        {
                            "status": i_sol.get_error_status(),
                            "solution": i_sol.get_error_message(),
                        }
                    )
                else:
                    res.append(create_solution(i_sol))
        elif sol is not None:
            if sol.get_error_status() != ErrorStatus.Success:
                raise get_solver_exception_type(
                    sol.get_error_status(), sol.get_error_message()
                )
            res = create_solution(sol)

        return notes, warnings, res, total_solve_time

    except (InputValidationError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (InputRuntimeError, OutOfMemoryError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
