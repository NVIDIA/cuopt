/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUOPT_CONSTANTS_H
#define CUOPT_CONSTANTS_H

#ifdef __cplusplus
#include <limits>
#else
#include <math.h>
#endif

#define CUOPT_INSTANTIATE_FLOAT  0
#define CUOPT_INSTANTIATE_DOUBLE 1
#define CUOPT_INSTANTIATE_INT32  1
#define CUOPT_INSTANTIATE_INT64  0

/* @brief LP/MIP parameter string constants */
#define CUOPT_ABSOLUTE_DUAL_TOLERANCE     "absolute_dual_tolerance"
#define CUOPT_RELATIVE_DUAL_TOLERANCE     "relative_dual_tolerance"
#define CUOPT_ABSOLUTE_PRIMAL_TOLERANCE   "absolute_primal_tolerance"
#define CUOPT_RELATIVE_PRIMAL_TOLERANCE   "relative_primal_tolerance"
#define CUOPT_ABSOLUTE_GAP_TOLERANCE      "absolute_gap_tolerance"
#define CUOPT_RELATIVE_GAP_TOLERANCE      "relative_gap_tolerance"
#define CUOPT_INFEASIBILITY_DETECTION     "infeasibility_detection"
#define CUOPT_STRICT_INFEASIBILITY        "strict_infeasibility"
#define CUOPT_PRIMAL_INFEASIBLE_TOLERANCE "primal_infeasible_tolerance"
#define CUOPT_DUAL_INFEASIBLE_TOLERANCE   "dual_infeasible_tolerance"
#define CUOPT_ITERATION_LIMIT             "iteration_limit"
#define CUOPT_TIME_LIMIT                  "time_limit"
#define CUOPT_PDLP_SOLVER_MODE            "pdlp_solver_mode"
#define CUOPT_METHOD                      "method"
#define CUOPT_PER_CONSTRAINT_RESIDUAL     "per_constraint_residual"
#define CUOPT_SAVE_BEST_PRIMAL_SO_FAR     "save_best_primal_so_far"
#define CUOPT_FIRST_PRIMAL_FEASIBLE       "first_primal_feasible"
#define CUOPT_LOG_FILE                    "log_file"
#define CUOPT_LOG_TO_CONSOLE              "log_to_console"
#define CUOPT_CROSSOVER                   "crossover"
#define CUOPT_FOLDING                     "folding"
#define CUOPT_AUGMENTED                   "augmented"
#define CUOPT_DUALIZE                     "dualize"
#define CUOPT_ORDERING                    "ordering"
#define CUOPT_BARRIER_DUAL_INITIAL_POINT  "barrier_dual_initial_point"
#define CUOPT_ELIMINATE_DENSE_COLUMNS     "eliminate_dense_columns"
#define CUOPT_CUDSS_DETERMINISTIC         "cudss_deterministic"
#define CUOPT_PRESOLVE                    "presolve"
#define CUOPT_DUAL_POSTSOLVE              "dual_postsolve"
#define CUOPT_MIP_ABSOLUTE_TOLERANCE      "mip_absolute_tolerance"
#define CUOPT_MIP_RELATIVE_TOLERANCE      "mip_relative_tolerance"
#define CUOPT_MIP_INTEGRALITY_TOLERANCE   "mip_integrality_tolerance"
#define CUOPT_MIP_ABSOLUTE_GAP            "mip_absolute_gap"
#define CUOPT_MIP_RELATIVE_GAP            "mip_relative_gap"
#define CUOPT_MIP_HEURISTICS_ONLY         "mip_heuristics_only"
#define CUOPT_MIP_SCALING                 "mip_scaling"
#define CUOPT_MIP_PRESOLVE                "mip_presolve"
#define CUOPT_SOLUTION_FILE               "solution_file"
#define CUOPT_NUM_CPU_THREADS             "num_cpu_threads"
#define CUOPT_USER_PROBLEM_FILE           "user_problem_file"

/* @brief LP/MIP termination status constants */
#define CUOPT_TERIMINATION_STATUS_NO_TERMINATION   0
#define CUOPT_TERIMINATION_STATUS_OPTIMAL          1
#define CUOPT_TERIMINATION_STATUS_INFEASIBLE       2
#define CUOPT_TERIMINATION_STATUS_UNBOUNDED        3
#define CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT  4
#define CUOPT_TERIMINATION_STATUS_TIME_LIMIT       5
#define CUOPT_TERIMINATION_STATUS_NUMERICAL_ERROR  6
#define CUOPT_TERIMINATION_STATUS_PRIMAL_FEASIBLE  7
#define CUOPT_TERIMINATION_STATUS_FEASIBLE_FOUND   8
#define CUOPT_TERIMINATION_STATUS_CONCURRENT_LIMIT 9

/* @brief The objective sense constants */
#define CUOPT_MINIMIZE 1
#define CUOPT_MAXIMIZE -1

/* @brief The constraint sense constants */
#define CUOPT_LESS_THAN    'L'
#define CUOPT_GREATER_THAN 'G'
#define CUOPT_EQUAL        'E'

/* @brief The variable type constants */
#define CUOPT_CONTINUOUS 'C'
#define CUOPT_INTEGER    'I'

/* @brief The infinity constant */
#ifdef __cplusplus
// Use the C++11 standard library for INFINITY
#define CUOPT_INFINITY std::numeric_limits<double>::infinity()
#else
// Use the C99 standard macro for INFINITY
#define CUOPT_INFINITY INFINITY
#endif

#define CUOPT_PDLP_SOLVER_MODE_STABLE1     0
#define CUOPT_PDLP_SOLVER_MODE_STABLE2     1
#define CUOPT_PDLP_SOLVER_MODE_METHODICAL1 2
#define CUOPT_PDLP_SOLVER_MODE_FAST1       3
#define CUOPT_PDLP_SOLVER_MODE_STABLE3     4

#define CUOPT_METHOD_CONCURRENT   0
#define CUOPT_METHOD_PDLP         1
#define CUOPT_METHOD_DUAL_SIMPLEX 2
#define CUOPT_METHOD_BARRIER      3

/* @brief Status codes constants */
#define CUOPT_SUCCESS          0
#define CUOPT_INVALID_ARGUMENT 1
#define CUOPT_MPS_FILE_ERROR   2
#define CUOPT_MPS_PARSE_ERROR  3
#define CUOPT_VALIDATION_ERROR 4
#define CUOPT_OUT_OF_MEMORY    5
#define CUOPT_RUNTIME_ERROR    6

#endif  // CUOPT_CONSTANTS_H
