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

#ifndef CUOPT_C_API_H
#define CUOPT_C_API_H

#include <cuopt/linear_programming/constants.h>

#include <stdint.h>

#ifdef __cplusplus

extern "C" {
#endif

/**
 * @brief A ``cuOptOptimizationProblem`` object contains a representation of
 * an LP or MIP. It is created by ``cuOptCreateProblem`` or ``cuOptCreateRangedProblem``.
 * It is passed to ``cuOptSolve``. It should be destroyed using ``cuOptDestroyProblem``.
 */
typedef void* cuOptOptimizationProblem;

/**
 * @brief A ``cuOptSolverSettings`` object contains parameter settings and other information
 * for an LP or MIP solve. It is created by ``cuOptCreateSolverSettings``. It is passed to
 * ``cuOptSolve``. It should be destroyed using ``cuOptDestroySolverSettings``.
 */
typedef void* cuOptSolverSettings;

/**
 * @brief A ``cuOptSolution`` object contains the solution to an LP or MIP. It is created by
 * ``cuOptSolve``. It should be destroyed using ``cuOptDestroySolution``.
 */
typedef void* cuOptSolution;

#if CUOPT_INSTANTIATE_FLOAT

/**
 * @brief The type of the floating point number used by the solver. Use ``cuOptGetFloatSize``
 * to get the number of bytes in the floating point type.
 */
typedef float cuopt_float_t;

#endif

#if CUOPT_INSTANTIATE_DOUBLE
/**
 * @brief The type of the floating point number used by the solver. Use ``cuOptGetFloatSize``
 * to get the size of the floating point type.
 */
typedef double cuopt_float_t;
#endif

#if CUOPT_INSTANTIATE_INT32
/**
 * @brief The type of the integer number used by the solver. Use ``cuOptGetIntSize``
 * to get the size of the integer type.
 */
typedef int32_t cuopt_int_t;
#endif

#if CUOPT_INSTANTIATE_INT64
/**
 * @brief The type of the integer number used by the solver. Use ``cuOptGetIntSize``
 * to get the size of the integer type.
 */
typedef int64_t cuopt_int_t;
#endif

/**
 * @brief Get the size of the float type.
 *
 * @return The size in bytes of the float type.
 */
int8_t cuOptGetFloatSize();

/** @brief Get the size of the integer type used by the library.
 * @return The size of the integer type in bytes.
 */
int8_t cuOptGetIntSize();

/**
 * @brief Get the version of the library.
 *
 * @param[out] version_major - A pointer to a cuopt_int_t that will contain the major version
 * number.
 * @param[out] version_minor - A pointer to a cuopt_int_t that will contain the minor version
 * number.
 * @param[out] version_patch - A pointer to a cuopt_int_t that will contain the patch version
 * number.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetVersion(cuopt_int_t* version_major,
                            cuopt_int_t* version_minor,
                            cuopt_int_t* version_patch);

/**
 * @brief Read an optimization problem from an MPS file.
 *
 * @param[in] filename - The path to the MPS file.
 *
 * @param[out] problem_ptr - A pointer to a cuOptOptimizationProblem. On output
 *  the problem will be created and initialized with the data from the MPS file
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptReadProblem(const char* filename, cuOptOptimizationProblem* problem_ptr);

/** @brief Create an optimization problem of the form
 *
 * @verbatim
 *                minimize/maximize  cᵀx + offset
 *                  subject to       A x {=, ≤, ≥} b
 *                                   l ≤ x ≤ u
 *                                   x_i integer for some i
 * @endverbatim
 *
 * @param[in] num_constraints The number of constraints
 * @param[in] num_variables The number of variables
 * @param[in] objective_sense The objective sense (CUOPT_MINIMIZE for
 *            minimization or CUOPT_MAXIMIZE for maximization)
 * @param[in] objective_offset An offset to add to the linear objective
 * @param[in] objective_coefficients A pointer to an array of type cuopt_float_t
 *            of size num_variables containing the coefficients of the linear objective
 * @param[in] constraint_matrix_row_offsets A pointer to an array of type
 *            cuopt_int_t of size num_constraints + 1. constraint_matrix_row_offsets[i] is the
 *            index of the first non-zero element of the i-th constraint in
 *            constraint_matrix_column_indices and constraint_matrix_coefficent_values. This is
 *            part of the compressed sparse row representation of the constraint matrix
 * @param[in] constraint_matrix_column_indices A pointer to an array of type
 *            cuopt_int_t of size constraint_matrix_row_offsets[num_constraints] containing
 *            the column indices of the non-zero elements of the constraint matrix. This is
 *            part of the compressed sparse row representation of the constraint matrix
 * @param[in] constraint_matrix_coefficent_values A pointer to an array of type
 *            cuopt_float_t of size constraint_matrix_row_offsets[num_constraints] containing
 *            the values of the non-zero elements of the constraint matrix. This is
 *            part of the compressed sparse row representation of the constraint matrix
 * @param[in] constraint_sense A pointer to an array of type char of size
 *            num_constraints containing the sense of the constraints (CUOPT_LESS_THAN,
 *            CUOPT_GREATER_THAN, or CUOPT_EQUAL)
 * @param[in] rhs A pointer to an array of type cuopt_float_t of size num_constraints
 *            containing the right-hand side of the constraints
 * @param[in] lower_bounds A pointer to an array of type cuopt_float_t of size num_variables
 *            containing the lower bounds of the variables
 * @param[in] upper_bounds A pointer to an array of type cuopt_float_t of size num_variables
 *            containing the upper bounds of the variables
 * @param[in] variable_types A pointer to an array of type char of size num_variables
 *            containing the types of the variables (CUOPT_CONTINUOUS or CUOPT_INTEGER)
 * @param[out] problem_ptr Pointer to store the created optimization problem
 * @return CUOPT_SUCCESS if successful, CUOPT_ERROR otherwise
 */
cuopt_int_t cuOptCreateProblem(cuopt_int_t num_constraints,
                               cuopt_int_t num_variables,
                               cuopt_int_t objective_sense,
                               cuopt_float_t objective_offset,
                               const cuopt_float_t* objective_coefficients,
                               const cuopt_int_t* constraint_matrix_row_offsets,
                               const cuopt_int_t* constraint_matrix_column_indices,
                               const cuopt_float_t* constraint_matrix_coefficent_values,
                               const char* constraint_sense,
                               const cuopt_float_t* rhs,
                               const cuopt_float_t* lower_bounds,
                               const cuopt_float_t* upper_bounds,
                               const char* variable_types,
                               cuOptOptimizationProblem* problem_ptr);

/** @brief Create an optimization problem of the form *
 * @verbatim
 *                minimize/maximize  c^T x + offset
 *                  subject to       bl <= A*x <= bu
 *                                   l <= x <= u
 *                                   x_i integer for some i
 * @endverbatim
 *
 * @param[in] num_constraints - The number of constraints.
 *
 * @param[in] num_variables - The number of variables.
 *
 * @param[in] objective_sense - The objective sense (CUOPT_MINIMIZE for
 *  minimization or CUOPT_MAXIMIZE for maximization)
 *
 * @param[in] objective_offset - An offset to add to the linear objective.
 *
 * @param[in] objective_coefficients - A pointer to an array of type cuopt_float_t
 *  of size num_variables containing the coefficients of the linear objective.
 *
 * @param[in] constraint_matrix_row_offsets - A pointer to an array of type
 *  cuopt_int_t of size num_constraints + 1. constraint_matrix_row_offsets[i] is the
 *  index of the first non-zero element of the i-th constraint in
 *  constraint_matrix_column_indices and constraint_matrix_coefficients.
 *
 * @param[in] constraint_matrix_column_indices - A pointer to an array of type
 *  cuopt_int_t of size constraint_matrix_row_offsets[num_constraints] containing
 *  the column indices of the non-zero elements of the constraint matrix.
 *
 * @param[in] constraint_matrix_coefficients - A pointer to an array of type
 *  cuopt_float_t of size constraint_matrix_row_offsets[num_constraints] containing
 *  the values of the non-zero elements of the constraint matrix.
 *
 * @param[in] constraint_lower_bounds - A pointer to an array of type
 *  cuopt_float_t of size num_constraints containing the lower bounds of the constraints.
 *
 * @param[in] constraint_upper_bounds - A pointer to an array of type
 *  cuopt_float_t of size num_constraints containing the upper bounds of the constraints.
 *
 * @param[in] variable_lower_bounds - A pointer to an array of type
 *  cuopt_float_t of size num_variables containing the lower bounds of the variables.
 *
 * @param[in] variable_upper_bounds - A pointer to an array of type
 *  cuopt_float_t of size num_variables containing the upper bounds of the variables.
 *
 * @param[in] variable_types - A pointer to an array of type char of size
 *  num_variables containing the types of the variables (CUOPT_CONTINUOUS or
 *  CUOPT_INTEGER).
 *
 * @param[out] problem_ptr - A pointer to a cuOptOptimizationProblem.
 * On output the problem will be created and initialized with the provided data.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptCreateRangedProblem(cuopt_int_t num_constraints,
                                     cuopt_int_t num_variables,
                                     cuopt_int_t objective_sense,
                                     cuopt_float_t objective_offset,
                                     const cuopt_float_t* objective_coefficients,
                                     const cuopt_int_t* constraint_matrix_row_offsets,
                                     const cuopt_int_t* constraint_matrix_column_indices,
                                     const cuopt_float_t* constraint_matrix_coefficients,
                                     const cuopt_float_t* constraint_lower_bounds,
                                     const cuopt_float_t* constraint_upper_bounds,
                                     const cuopt_float_t* variable_lower_bounds,
                                     const cuopt_float_t* variable_upper_bounds,
                                     const char* variable_types,
                                     cuOptOptimizationProblem* problem_ptr);

/** @brief Destroy an optimization problem
 *
 * @param[in, out] problem_ptr - A pointer to a cuOptOptimizationProblem. On
 *  output the problem will be destroyed, and the pointer will be set to NULL.
 */
void cuOptDestroyProblem(cuOptOptimizationProblem* problem_ptr);

/** @brief Get the number of constraints of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] num_constraints_ptr - A pointer to a cuopt_int_t that will contain the
 *  number of constraints on output.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetNumConstraints(cuOptOptimizationProblem problem,
                                   cuopt_int_t* num_constraints_ptr);

/** @brief Get the number of variables of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] num_variables_ptr - A pointer to a cuopt_int_t that will contain the
 *  number of variables on output.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetNumVariables(cuOptOptimizationProblem problem, cuopt_int_t* num_variables_ptr);

/** @brief Get the objective sense of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] objective_sense_ptr - A pointer to a cuopt_int_t that on output
 *  will contain the objective sense.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetObjectiveSense(cuOptOptimizationProblem problem,
                                   cuopt_int_t* objective_sense_ptr);

/** @brief Get the objective offset of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] objective_offset_ptr - A pointer to a cuopt_float_t that on output
 *  will contain the objective offset.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetObjectiveOffset(cuOptOptimizationProblem problem,
                                    cuopt_float_t* objective_offset_ptr);

/** @brief Get the objective coefficients of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] objective_coefficients_ptr - A pointer to an array of type
 *  cuopt_float_t of size num_variables that on output will contain the objective
 *  coefficients.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetObjectiveCoefficients(cuOptOptimizationProblem problem,
                                          cuopt_float_t* objective_coefficients_ptr);

/** @brief Get the number of non-zero elements in the constraint matrix of an
 *  optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] num_non_zeros_ptr - A pointer to a cuopt_int_t that on output
 *  will contain the number of non-zeros in the constraint matrix.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetNumNonZeros(cuOptOptimizationProblem problem, cuopt_int_t* num_non_zeros_ptr);

/** @brief Get the constraint matrix of an optimization problem in compressed sparse row format.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] constraint_matrix_row_offsets_ptr - A pointer to an array of type
 *  cuopt_int_t of size num_constraints + 1 that on output will contain the row
 *  offsets of the constraint matrix.
 *
 * @param[out] constraint_matrix_column_indices_ptr - A pointer to an array of type
 *  cuopt_int_t of size equal to the number of nonzeros that on output will contain the
 *  column indices of the non-zero entries of the constraint matrix.
 *
 * @param[out] constraint_matrix_coefficients_ptr - A pointer to an array of type
 *  cuopt_float_t of size equal to the number of nonzeros that on output will contain the
 *  coefficients of the non-zero entries of the constraint matrix.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetConstraintMatrix(cuOptOptimizationProblem problem,
                                     cuopt_int_t* constraint_matrix_row_offsets_ptr,
                                     cuopt_int_t* constraint_matrix_column_indices_ptr,
                                     cuopt_float_t* constraint_matrix_coefficients_ptr);

/** @brief Get the constraint sense of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] constraint_sense_ptr - A pointer to an array of type char of size
 *  num_constraints that on output will contain the sense of the constraints.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetConstraintSense(cuOptOptimizationProblem problem, char* constraint_sense_ptr);

/** @brief Get the right-hand side of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] rhs_ptr - A pointer to an array of type cuopt_float_t of size
 *  num_constraints that on output will contain the right-hand side of the constraints.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetConstraintRightHandSide(cuOptOptimizationProblem problem,
                                            cuopt_float_t* rhs_ptr);

/** @brief Get the lower bounds of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] lower_bounds_ptr - A pointer to an array of type cuopt_float_t of size
 *  num_constraints that on output will contain the lower bounds of the constraints.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetConstraintLowerBounds(cuOptOptimizationProblem problem,
                                          cuopt_float_t* lower_bounds_ptr);

/** @brief Get the upper bounds of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] upper_bounds_ptr - A pointer to an array of type cuopt_float_t of size
 *  num_constraints that on output will contain the upper bounds of the constraints.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetConstraintUpperBounds(cuOptOptimizationProblem problem,
                                          cuopt_float_t* upper_bounds_ptr);

/** @brief Get the lower bounds of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] lower_bounds_ptr - A pointer to an array of type cuopt_float_t of size
 *  num_variables that on output will contain the lower bounds of the variables.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetVariableLowerBounds(cuOptOptimizationProblem problem,
                                        cuopt_float_t* lower_bounds_ptr);

/** @brief Get the upper bounds of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] upper_bounds_ptr - A pointer to an array of type cuopt_float_t of size
 *  num_variables that on output will contain the upper bounds of the variables.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetVariableUpperBounds(cuOptOptimizationProblem problem,
                                        cuopt_float_t* upper_bounds_ptr);

/** @brief Get the variable types of an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] variable_types_ptr - A pointer to an array of type char of size
 *  num_variables that on output will contain the types of the variables
 *  (CUOPT_CONTINUOUS or CUOPT_INTEGER).
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetVariableTypes(cuOptOptimizationProblem problem, char* variable_types_ptr);

/** @brief Create a solver settings object.
 *
 * @param[out] settings_ptr - A pointer to a cuOptSolverSettings object. On output
 *  the solver settings will be created and initialized.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptCreateSolverSettings(cuOptSolverSettings* settings_ptr);

/** @brief Destroy a solver settings object.
 *
 * @param[in, out] settings_ptr - A pointer to a cuOptSolverSettings object. On output
 *  the solver settings will be destroyed and the pointer will be set to NULL.
 */
void cuOptDestroySolverSettings(cuOptSolverSettings* settings_ptr);

/** @brief Set a parameter of a solver settings object.
 *
 * @param[in] settings - The solver settings object.
 *
 * @param[in] parameter_name - The name of the parameter to set.
 *
 * @param[in] parameter_value - The value of the parameter to set.
 */
cuopt_int_t cuOptSetParameter(cuOptSolverSettings settings,
                              const char* parameter_name,
                              const char* parameter_value);

/** @brief Get a parameter of a solver settings object.
 *
 * @param[in] settings - The solver settings object.
 *
 * @param[in] parameter_name - The name of the parameter to get.
 *
 * @param[in] parameter_value_size - The size of the parameter value buffer.
 *
 * @param[out] parameter_value - A pointer to an array of characters that on output will contain the
 *  value of the parameter.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetParameter(cuOptSolverSettings settings,
                              const char* parameter_name,
                              cuopt_int_t parameter_value_size,
                              char* parameter_value);

/** @brief Set an integer parameter of a solver settings object.
 *
 * @param[in] settings - The solver settings object.
 *
 * @param[in] parameter_name - The name of the parameter to set.
 *
 * @param[in] parameter_value - The value of the parameter to set.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptSetIntegerParameter(cuOptSolverSettings settings,
                                     const char* parameter_name,
                                     cuopt_int_t parameter_value);

/** @brief Get an integer parameter of a solver settings object.
 *
 * @param[in] settings - The solver settings object.
 *
 * @param[in] parameter_name - The name of the parameter to get.
 *
 * @param[out] parameter_value - A pointer to a cuopt_int_t that on output will contain the
 *  value of the parameter.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetIntegerParameter(cuOptSolverSettings settings,
                                     const char* parameter_name,
                                     cuopt_int_t* parameter_value);

/** @brief Set a float parameter of a solver settings object.
 *
 * @param[in] settings - The solver settings object.
 *
 * @param[in] parameter_name - The name of the parameter to set.
 *
 * @param[in] parameter_value - The value of the parameter to set.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptSetFloatParameter(cuOptSolverSettings settings,
                                   const char* parameter_name,
                                   cuopt_float_t parameter_value);

/** @brief Get a float parameter of a solver settings object.
 *
 * @param[in] settings - The solver settings object.
 *
 * @param[in] parameter_name - The name of the parameter to get.
 *
 * @param[out] parameter_value - A pointer to a cuopt_float_t that on output will contain the
 *  value of the parameter.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetFloatParameter(cuOptSolverSettings settings,
                                   const char* parameter_name,
                                   cuopt_float_t* parameter_value);

/** @brief Check if an optimization problem is a mixed integer programming problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[out] is_mip_ptr - A pointer to a cuopt_int_t that on output will be 0 if the problem
 * contains only continuous variables, or 1 if the problem contains integer variables.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptIsMIP(cuOptOptimizationProblem problem, cuopt_int_t* is_mip_ptr);

/** @brief Solve an optimization problem.
 *
 * @param[in] problem - The optimization problem.
 *
 * @param[in] settings - The solver settings.
 *
 * @param[out] solution_ptr - A pointer to a cuOptSolution object. On output
 *  the solution will be created.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptSolve(cuOptOptimizationProblem problem,
                       cuOptSolverSettings settings,
                       cuOptSolution* solution_ptr);

/** @brief Destroy a solution object.
 *
 * @param[in, out] solution_ptr - A pointer to a cuOptSolution object. On output
 *  the solution will be destroyed and the pointer will be set to NULL.
 */
void cuOptDestroySolution(cuOptSolution* solution_ptr);

/** @brief Get the termination reason of an optimization problem.
 *
 * @param[in] solution - The solution object.
 *
 * @param[out] termination_reason_ptr - A pointer to a cuopt_int_t that on output will contain the
 *  termination reason.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetTerminationStatus(cuOptSolution solution, cuopt_int_t* termination_status_ptr);

/* @brief Get the error status of a solution object.
 *
 * @param[in] solution - The solution object.
 *
 * @param[out] error_status_ptr - A pointer to a cuopt_int_t that on output will contain the
 *  error status.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetErrorStatus(cuOptSolution solution, cuopt_int_t* error_status_ptr);

/* @brief Get the error string of a solution object.
 *
 * @param[in] solution - The solution object.
 *
 * @param[out] error_string_ptr - A pointer to a char that on output will contain the
 *  error string.
 *
 * @param[in] error_string_size - Size of the char buffer/
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetErrorString(cuOptSolution solution,
                                char* error_string_ptr,
                                cuopt_int_t error_string_size);

/* @brief Get the solution of an optimization problem.
 *
 * @param[in] solution - The solution object.
 *
 * @param[in, out] solution_values - A pointer to an array of type cuopt_float_t of size
 * num_variables that will contain the solution values.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetPrimalSolution(cuOptSolution solution, cuopt_float_t* solution_values);

/** @brief Get the objective value of an optimization problem.
 *
 * @param[in] solution - The solution object.
 *
 * @param[in,out] objective_value_ptr - A pointer to a cuopt_float_t that will contain the objective
 * value.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetObjectiveValue(cuOptSolution solution, cuopt_float_t* objective_value_ptr);

/** @brief Get the solve time of an optimization problem.
 *
 * @param[in] solution - The solution object.
 *
 * @param[in,out] solve_time_ptr - A pointer to a cuopt_float_t that will contain the solve time.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetSolveTime(cuOptSolution solution, cuopt_float_t* solve_time_ptr);

/** @brief Get the relative MIP gap of an optimization problem.
 *
 * @param[in] solution - The solution object.
 *
 * @param[in, out] mip_gap_ptr - A pointer to a cuopt_float_t that will contain the relative MIP
 * gap.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetMIPGap(cuOptSolution solution, cuopt_float_t* mip_gap_ptr);

/** @brief Get the solution bound of an optimization problem.
 *
 * @param[in] solution - The solution object.
 *
 * @param[in, out] solution_bound_ptr - A pointer to a cuopt_float_t that will contain the solution
 * bound.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetSolutionBound(cuOptSolution solution, cuopt_float_t* solution_bound_ptr);

/** @brief Get the dual solution of an optimization problem.
 *
 * @param[in] solution - The solution object.
 *
 * @param[in, out] dual_solution_ptr - A pointer to an array of type cuopt_float_t of size
 * num_constraints that will contain the dual solution.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetDualSolution(cuOptSolution solution, cuopt_float_t* dual_solution_ptr);

/** @brief Get the dual objective value of an optimization problem.
 *
 * @param[in] solution - The solution object.
 *
 * @param[in, out] dual_objective_value_ptr - A pointer to a cuopt_float_t that will contain the
 * dual objective value.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetDualObjectiveValue(cuOptSolution solution,
                                       cuopt_float_t* dual_objective_value_ptr);

/** @brief Get the reduced costs of an optimization problem.
 *
 * @param[in] solution - The solution object.
 *
 * @param[in,out] reduced_cost_ptr - A pointer to an array of type cuopt_float_t of size
 * num_variables that will contain the reduced cost.
 *
 * @return A status code indicating success or failure.
 */
cuopt_int_t cuOptGetReducedCosts(cuOptSolution solution, cuopt_float_t* reduced_cost_ptr);

#ifdef __cplusplus
}
#endif

#endif  // CUOPT_C_API_H
