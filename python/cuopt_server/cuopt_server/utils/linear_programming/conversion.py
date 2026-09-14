# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from cuopt import linear_programming
from cuopt.linear_programming.solver.solver_parameters import solver_params


def ignored_warning(field):
    return f"solver config {field} ignored in the cuopt service"


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
        for param in solver_params:
            param_value = None
            if param.endswith("tolerance"):
                param_value = getattr(solver_config.tolerances, param, None)
            else:
                param_value = getattr(solver_config, param, None)
            if param_value is not None and param_value != "":
                solver_settings.set_parameter(param, param_value)

    if LP_data.solver_config is not None:
        solver_config = LP_data.solver_config

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
            solver_settings.set_parameter("time_limit", time_limit)

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
            solver_settings.set_parameter("iteration_limit", iteration_limit)

        if warmstart_data is not None:
            solver_settings.set_pdlp_warm_start_data(warmstart_data)

        if solver_config.user_problem_file != "":
            warnings.append(ignored_warning("user_problem_file"))

        if solver_config.solution_file != "":
            warnings.append(ignored_warning("solution_file"))

    return warnings, solver_settings
