/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt/linear_programming/cuopt_c.h>

void cuopt_java_release_settings_state(cuOptSolverSettings settings);

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

cuopt_int_t cuOptSetPDLPWarmStartData(cuOptSolverSettings settings,
                                      const cuopt_float_t* current_primal_solution,
                                      const cuopt_float_t* current_dual_solution,
                                      const cuopt_float_t* initial_primal_average,
                                      const cuopt_float_t* initial_dual_average,
                                      const cuopt_float_t* current_ATY,
                                      const cuopt_float_t* sum_primal_solutions,
                                      const cuopt_float_t* sum_dual_solutions,
                                      const cuopt_float_t* last_restart_duality_gap_primal_solution,
                                      const cuopt_float_t* last_restart_duality_gap_dual_solution,
                                      cuopt_int_t primal_size,
                                      cuopt_int_t dual_size,
                                      cuopt_float_t initial_primal_weight,
                                      cuopt_float_t initial_step_size,
                                      cuopt_int_t total_pdlp_iterations,
                                      cuopt_int_t total_pdhg_iterations,
                                      cuopt_float_t last_candidate_kkt_score,
                                      cuopt_float_t last_restart_kkt_score,
                                      cuopt_float_t sum_solution_weight,
                                      cuopt_int_t iterations_since_last_restart);

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
cuopt_int_t cuOptHasPDLPWarmStartData(cuOptSolution solution, cuopt_int_t* has_warm_start_ptr);
cuopt_int_t cuOptGetPDLPWarmStartVectorSize(cuOptSolution solution,
                                            cuopt_int_t field_id,
                                            cuopt_int_t* size_ptr);
cuopt_int_t cuOptGetPDLPWarmStartVector(cuOptSolution solution,
                                        cuopt_int_t field_id,
                                        cuopt_float_t* values);
cuopt_int_t cuOptGetPDLPWarmStartScalar(cuOptSolution solution,
                                        cuopt_int_t field_id,
                                        cuopt_float_t* value_ptr);
cuopt_int_t cuOptGetPDLPWarmStartInteger(cuOptSolution solution,
                                         cuopt_int_t field_id,
                                         cuopt_int_t* value_ptr);
}

enum cuopt_java_pdlp_warm_start_field : cuopt_int_t {
  CUOPT_JAVA_PDLP_WARM_START_CURRENT_PRIMAL_SOLUTION                  = 0,
  CUOPT_JAVA_PDLP_WARM_START_CURRENT_DUAL_SOLUTION                    = 1,
  CUOPT_JAVA_PDLP_WARM_START_INITIAL_PRIMAL_AVERAGE                   = 2,
  CUOPT_JAVA_PDLP_WARM_START_INITIAL_DUAL_AVERAGE                     = 3,
  CUOPT_JAVA_PDLP_WARM_START_CURRENT_ATY                              = 4,
  CUOPT_JAVA_PDLP_WARM_START_SUM_PRIMAL_SOLUTIONS                     = 5,
  CUOPT_JAVA_PDLP_WARM_START_SUM_DUAL_SOLUTIONS                       = 6,
  CUOPT_JAVA_PDLP_WARM_START_LAST_RESTART_DUALITY_GAP_PRIMAL_SOLUTION = 7,
  CUOPT_JAVA_PDLP_WARM_START_LAST_RESTART_DUALITY_GAP_DUAL_SOLUTION   = 8,
  CUOPT_JAVA_PDLP_WARM_START_INITIAL_PRIMAL_WEIGHT                    = 9,
  CUOPT_JAVA_PDLP_WARM_START_INITIAL_STEP_SIZE                        = 10,
  CUOPT_JAVA_PDLP_WARM_START_TOTAL_PDLP_ITERATIONS                    = 11,
  CUOPT_JAVA_PDLP_WARM_START_TOTAL_PDHG_ITERATIONS                    = 12,
  CUOPT_JAVA_PDLP_WARM_START_LAST_CANDIDATE_KKT_SCORE                 = 13,
  CUOPT_JAVA_PDLP_WARM_START_LAST_RESTART_KKT_SCORE                   = 14,
  CUOPT_JAVA_PDLP_WARM_START_SUM_SOLUTION_WEIGHT                      = 15,
  CUOPT_JAVA_PDLP_WARM_START_ITERATIONS_SINCE_LAST_RESTART            = 16
};
