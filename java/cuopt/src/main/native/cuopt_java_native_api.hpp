/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt/mathematical_optimization/cuopt_c.h>

extern "C" {

cuopt_int_t cuOptLoadParametersFromFile(cuOptSolverSettings settings, const char* path);
cuopt_int_t cuOptDumpParametersToFile(cuOptSolverSettings settings,
                                      const char* path,
                                      cuopt_int_t hyperparameters_only,
                                      cuopt_int_t* dumped_successfully);
cuopt_int_t cuOptGetNumSolverParameters(cuopt_int_t* num_parameters_ptr);
cuopt_int_t cuOptGetSolverParameterName(cuopt_int_t index,
                                        cuopt_int_t parameter_name_size,
                                        char* parameter_name);

cuopt_int_t cuOptSolutionIsMIP(cuOptSolution solution, cuopt_int_t* is_mip_ptr);
cuopt_int_t cuOptGetLPSolverStats(cuOptSolution solution,
                                  cuopt_float_t* primal_residual_ptr,
                                  cuopt_float_t* dual_residual_ptr,
                                  cuopt_float_t* gap_ptr,
                                  cuopt_int_t* num_iterations_ptr,
                                  cuopt_int_t* solved_by_ptr);
cuopt_int_t cuOptGetMIPSolverStats(cuOptSolution solution,
                                   cuopt_float_t* presolve_time_ptr,
                                   cuopt_float_t* max_constraint_violation_ptr,
                                   cuopt_float_t* max_int_violation_ptr,
                                   cuopt_float_t* max_variable_bound_violation_ptr,
                                   cuopt_int_t* num_nodes_ptr,
                                   cuopt_int_t* num_simplex_iterations_ptr);
}
