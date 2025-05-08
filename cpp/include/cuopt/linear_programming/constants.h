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
#define CUOPT_ABSOLUTE_DUAL_TOLERANCE     "AbsoluteDualTolerance"
#define CUOPT_RELATIVE_DUAL_TOLERANCE     "RelativeDualTolerance"
#define CUOPT_ABSOLUTE_PRIMAL_TOLERANCE   "AbsolutePrimalTolerance"
#define CUOPT_RELATIVE_PRIMAL_TOLERANCE   "RelativePrimalTolerance"
#define CUOPT_ABSOLUTE_GAP_TOLERANCE      "AbsoluteGapTolerance"
#define CUOPT_RELATIVE_GAP_TOLERANCE      "RelativeGapTolerance"
#define CUOPT_INFEASIBILITY_DETECTION     "InfeasibilityDetection"
#define CUOPT_STRICT_INFEASIBILITY        "StrictInfeasibility"
#define CUOPT_PRIMAL_INFEASIBLE_TOLERANCE "PrimalInfeasibleTolerance"
#define CUOPT_DUAL_INFEASIBLE_TOLERANCE   "DualInfeasibleTolerance"
#define CUOPT_ITERATION_LIMIT             "IterationLimit"
#define CUOPT_TIME_LIMIT                  "TimeLimit"
#define CUOPT_PDLP_SOLVER_MODE            "PDLPSolverMode"
#define CUOPT_METHOD                      "Method"
#define CUOPT_PER_CONSTRAINT_RESIDUAL     "PerConstraintResidual"
#define CUOPT_SAVE_BEST_PRIMAL_SO_FAR     "SaveBestPrimalSoFar"
#define CUOPT_FIRST_PRIMAL_FEASIBLE       "FirstPrimalFeasible"
#define CUOPT_LOG_FILE                    "LogFile"
#define CUOPT_MIP_ABSOLUTE_TOLERANCE      "MIPAbsoluteTolerance"
#define CUOPT_MIP_RELATIVE_TOLERANCE      "MIPRelativeTolerance"
#define CUOPT_MIP_INTEGRALITY_TOLERANCE   "MIPIntegralityTolerance"
#define CUOPT_MIP_SCALING                 "MIPScaling"
#define CUOPT_MIP_HEURISTICS_ONLY         "MIPHeuristicsOnly"
#define CUOPT_NUM_CPU_THREADS             "NumCPUThreads"

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

#define CUOPT_METHOD_CONCURRENT   0
#define CUOPT_METHOD_PDLP         1
#define CUOPT_METHOD_DUAL_SIMPLEX 2

#endif  // CUOPT_CONSTANTS_H
