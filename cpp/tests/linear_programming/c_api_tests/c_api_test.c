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

#include "c_api_tests.h"

#include <cuopt/linear_programming/cuopt_c.h>

#include <stdio.h>
#include <stdlib.h>

#ifdef _cplusplus
#error "This file must be compiled as C code"
#endif


int check_problem(cuOptOptimizationProblem problem,
                  cuopt_int_t num_constraints,
                  cuopt_int_t num_variables,
                  cuopt_int_t nnz,
                  cuopt_int_t objective_sense,
                  cuopt_float_t objective_offset,
                  cuopt_float_t* objective_coefficients,
                  cuopt_int_t* row_offsets,
                  cuopt_int_t* column_indices,
                  cuopt_float_t* values,
                  char* constraint_sense,
                  cuopt_float_t* rhs,
                  cuopt_float_t* var_lower_bounds,
                  cuopt_float_t* var_upper_bounds,
                  char* variable_types);

const char* termination_status_to_string(cuopt_int_t termination_status)
{
  switch (termination_status) {
    case CUOPT_TERIMINATION_STATUS_OPTIMAL:
      return "Optimal";
    case CUOPT_TERIMINATION_STATUS_INFEASIBLE:
      return "Infeasible";
    case CUOPT_TERIMINATION_STATUS_UNBOUNDED:
      return "Unbounded";
    case CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT:
      return "Iteration limit";
    case CUOPT_TERIMINATION_STATUS_TIME_LIMIT:
      return "Time limit";
    case CUOPT_TERIMINATION_STATUS_NUMERICAL_ERROR:
      return "Numerical error";
    case CUOPT_TERIMINATION_STATUS_PRIMAL_FEASIBLE:
      return "Primal feasible";
    case CUOPT_TERIMINATION_STATUS_FEASIBLE_FOUND:
      return "Feasible found";
  }
}


int test_int_size() {
  return cuOptGetIntSize();
}

int test_float_size() {
  return cuOptGetFloatSize();
}

cuopt_int_t test_missing_file() {
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings = NULL;
  cuOptSolution solution = NULL;
  cuopt_int_t status = cuOptReadProblem("missing_file.mps", &problem);
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);
  return status;
}

cuopt_int_t test_bad_parameter_name() {
  cuOptSolverSettings settings = NULL;
  cuopt_int_t status;
  cuopt_int_t value;
  cuopt_float_t float_value;
#define BUFFER_SIZE 64
  char buffer[BUFFER_SIZE];

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings\n");
    goto DONE;
  }

  status = cuOptSetParameter(settings, "bad_parameter_name", "1");
  if (status != CUOPT_INVALID_ARGUMENT) {
    printf("Error: expected invalid argument error, but got %d\n", status);
    goto DONE;
  }

  status = cuOptGetParameter(settings, "bad_parameter_name", BUFFER_SIZE, buffer);
  if (status != CUOPT_INVALID_ARGUMENT) {
    printf("Error: expected invalid argument error, but got %d\n", status);
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, "bad_parameter_name", 1);
  if (status != CUOPT_INVALID_ARGUMENT) {
    printf("Error: expected invalid argument error, but got %d\n", status);
    goto DONE;
  }

  status = cuOptSetFloatParameter(settings, "bad_parameter_name", 1.0);
  if (status != CUOPT_INVALID_ARGUMENT) {
    printf("Error: expected invalid argument error, but got %d\n", status);
    goto DONE;
  }

  status = cuOptGetIntegerParameter(settings, "bad_parameter_name", &value);
  if (status != CUOPT_INVALID_ARGUMENT) {
    printf("Error: expected invalid argument error, but got %d\n", status);
    goto DONE;
  }

  status = cuOptGetFloatParameter(settings, "bad_parameter_name", &float_value);
  if (status != CUOPT_INVALID_ARGUMENT) {
    printf("Error: expected invalid argument error, but got %d\n", status);
    goto DONE;
  }

DONE:
  cuOptDestroySolverSettings(&settings);
  return status;
}

cuopt_int_t burglar_problem()
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
  /* Solve the burglar problem

  maximize sum_i value[i] * take[i]
  subject to sum_i weight[i] * take[i] <= max_weight
  take[i] binary for all i
  */

#define NUM_ITEMS       8
#define NUM_CONSTRAINTS 1
  cuopt_int_t num_items    = NUM_ITEMS;
  cuopt_float_t max_weight = 102;
  cuopt_float_t value[]    = {15, 100, 90, 60, 40, 15, 10, 1};
  cuopt_float_t weight[]   = {2, 20, 20, 30, 40, 30, 60, 10};

  cuopt_int_t num_variables   = NUM_ITEMS;
  cuopt_int_t num_constraints = NUM_CONSTRAINTS;
  cuopt_int_t nnz             = NUM_ITEMS;

  cuopt_int_t row_offsets[] = {0, NUM_ITEMS};
  cuopt_int_t column_indices[NUM_ITEMS];

  cuopt_float_t rhs[]         = {max_weight};
  char constraint_sense[] = {CUOPT_LESS_THAN};
  cuopt_float_t lower_bounds[NUM_ITEMS];
  cuopt_float_t upper_bounds[NUM_ITEMS];
  char variable_types[NUM_ITEMS];
  cuopt_int_t objective_sense    = CUOPT_MAXIMIZE;
  cuopt_float_t objective_offset = 0;
  cuopt_int_t is_mip;
  cuopt_int_t status;
  cuopt_float_t time;
  cuopt_int_t termination_status;
  cuopt_float_t objective_value;
#define BUFFER_SIZE 64
  char buffer[BUFFER_SIZE];

  for (cuopt_int_t j = 0; j < NUM_ITEMS; j++) {
    column_indices[j] = j;
  }

  for (cuopt_int_t j = 0; j < NUM_ITEMS; j++) {
    variable_types[j] = CUOPT_INTEGER;
    lower_bounds[j]   = 0;
    upper_bounds[j]   = 1;
  }

  status = cuOptCreateProblem(num_constraints,
                              num_variables,
                              objective_sense,
                              objective_offset,
                              value,
                              row_offsets,
                              column_indices,
                              weight,
                              constraint_sense,
                              rhs,
                              lower_bounds,
                              upper_bounds,
                              variable_types,
                              &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating optimization problem\n");
    goto DONE;
  }

  status = check_problem(problem,
                         num_constraints,
                         num_variables,
                         nnz,
                         objective_sense,
                         objective_offset,
                         value,
                         row_offsets,
                         column_indices,
                         weight,
                         constraint_sense,
                         rhs,
                         lower_bounds,
                         upper_bounds,
                         variable_types);
  if (status != CUOPT_SUCCESS) {
    printf("Error checking problem\n");
    goto DONE;
  }

  status = cuOptIsMIP(problem, &is_mip);
  if (status != CUOPT_SUCCESS) {
    printf("Error checking if problem is MIP\n");
    goto DONE;
  }
  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings\n");
    goto DONE;
  }
  status = cuOptGetParameter(settings, CUOPT_TIME_LIMIT, BUFFER_SIZE, buffer);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting time limit\n");
    goto DONE;
  }
  printf("Time limit: %s\n", buffer);

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem\n");
    goto DONE;
  }
  status = cuOptGetSolveTime(solution, &time);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solve time\n");
    goto DONE;
  }
  status = cuOptGetTerminationStatus(solution, &termination_status);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status\n");
    goto DONE;
  }
  if (termination_status != CUOPT_TERIMINATION_STATUS_OPTIMAL) {
    printf("Error: expected termination status to be %d, but got %d\n",
           CUOPT_TERIMINATION_STATUS_OPTIMAL,
           termination_status);
    status = -1;
    goto DONE;
  }
  status = cuOptGetObjectiveValue(solution, &objective_value);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value\n");
    goto DONE;
  }
  printf("Solve finished with termination status %s (%d) in %f seconds\n",
         termination_status_to_string(termination_status),
         termination_status,
         time);
  printf("Objective value: %f\n", objective_value);
DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

int solve_mps_file(const char* filename, double time_limit, double iteration_limit, int* termination_status_ptr, double* solve_time_ptr, int method)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings = NULL;
  cuOptSolution solution = NULL;
  cuopt_int_t status;
  cuopt_int_t is_mip;
  cuopt_int_t termination_status = -1;
  cuopt_float_t time;
  cuopt_float_t objective_value;
  printf("Reading problem from %s\n", filename);
  status = cuOptReadProblem(filename, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error reading problem\n");
    goto DONE;
  }
  status = cuOptIsMIP(problem, &is_mip);
  if (status != CUOPT_SUCCESS) {
    printf("Error checking if problem is MIP\n");
    goto DONE;
  };
  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings\n");
    goto DONE;
  }
  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, method);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting method\n");
    goto DONE;
  }
  status = cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, time_limit);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting time limit\n");
    goto DONE;
  }
  if (iteration_limit < CUOPT_INFINITY) {
    cuopt_int_t iteration_limit_int = (cuopt_int_t)iteration_limit;
    printf("Setting iteration limit to %d\n", iteration_limit_int);
    status = cuOptSetIntegerParameter(settings, CUOPT_ITERATION_LIMIT, iteration_limit_int);
    if (status != CUOPT_SUCCESS) {
      printf("Error setting iteration limit\n");
      goto DONE;
    }
  }
  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    #define ERROR_BUFFER_SIZE 1024
    char error_string[ERROR_BUFFER_SIZE];
    cuopt_int_t error_string_status = cuOptGetErrorString(solution, error_string, ERROR_BUFFER_SIZE);
    if (error_string_status != CUOPT_SUCCESS) {
      printf("Error getting error string\n");
      goto DONE;
    }
    printf("Error %d solving problem: %s\n", status, error_string);
    goto DONE;
  }
  status = cuOptGetSolveTime(solution, &time);
  if (solve_time_ptr) *solve_time_ptr = time;
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solve time\n");
    goto DONE;
  }
  status = cuOptGetTerminationStatus(solution, &termination_status);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status\n");
    goto DONE;
  }
  status = cuOptGetObjectiveValue(solution, &objective_value);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value\n");
    goto DONE;
  }
  printf("Solve finished with termination status %s (%d) in %f seconds\n",
         termination_status_to_string(termination_status),
         termination_status,
         time);
  printf("Objective value: %f\n", objective_value);
DONE:
  *termination_status_ptr = termination_status;
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

int check_problem(cuOptOptimizationProblem problem,
                  cuopt_int_t num_constraints,
                  cuopt_int_t num_variables,
                  cuopt_int_t nnz,
                  cuopt_int_t objective_sense,
                  cuopt_float_t objective_offset,
                  cuopt_float_t* objective_coefficients,
                  cuopt_int_t* row_offsets,
                  cuopt_int_t* column_indices,
                  cuopt_float_t* values,
                  char* constraint_sense,
                  cuopt_float_t* rhs,
                  cuopt_float_t* var_lower_bounds,
                  cuopt_float_t* var_upper_bounds,
                  char* variable_types)
{
  cuopt_int_t check_num_constraints;
  cuopt_int_t check_num_variables;
  cuopt_int_t check_nnz;
  cuopt_int_t check_objective_sense;
  cuopt_float_t check_objective_offset;
  cuopt_float_t* check_objective_coefficients;
  cuopt_int_t* check_row_offsets;
  cuopt_int_t* check_column_indices;
  cuopt_float_t* check_values;
  char* check_constraint_sense;
  cuopt_float_t* check_rhs;
  cuopt_float_t* check_var_lower_bounds;
  cuopt_float_t* check_var_upper_bounds;
  char* check_variable_types;
  cuopt_int_t status;
  check_objective_coefficients = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  check_row_offsets = (cuopt_int_t*)malloc((num_constraints + 1) * sizeof(cuopt_int_t));
  check_column_indices = (cuopt_int_t*)malloc(nnz * sizeof(cuopt_int_t));
  check_values = (cuopt_float_t*)malloc(nnz * sizeof(cuopt_float_t));
  check_constraint_sense = (char*)malloc(num_constraints * sizeof(char));
  check_rhs = (cuopt_float_t*)malloc(num_constraints * sizeof(cuopt_float_t));
  check_var_lower_bounds = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  check_var_upper_bounds = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  check_variable_types = (char*)malloc(num_variables * sizeof(char));

  status = cuOptGetNumConstraints(problem, &check_num_constraints);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting number of constraints\n");
    goto DONE;
  }
  if (check_num_constraints != num_constraints) {
    printf("Error: expected number of constraints to be %d, but got %d\n",
           num_constraints,
           check_num_constraints);
    status = -1;
    goto DONE;
  }

  status = cuOptGetNumVariables(problem, &check_num_variables);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting number of variables\n");
    goto DONE;
  }
  if (check_num_variables != num_variables) {
    printf("Error: expected number of variables to be %d, but got %d\n",
           num_variables,
           check_num_variables);
    status = -1;
    goto DONE;
  }

  status = cuOptGetNumNonZeros(problem, &check_nnz);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting number of non-zeros\n");
    goto DONE;
  }
  if (check_nnz != nnz) {
    printf("Error: expected number of non-zeros to be %d, but got %d\n", nnz, check_nnz);
    status = -1;
    goto DONE;
  }

  status = cuOptGetObjectiveSense(problem, &check_objective_sense);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective sense\n");
    goto DONE;
  }
  if (check_objective_sense != objective_sense) {
    printf("Error: expected objective sense to be %d, but got %d\n",
           objective_sense,
           check_objective_sense);
    status = -1;
    goto DONE;
  }

  status = cuOptGetObjectiveOffset(problem, &check_objective_offset);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective offset\n");
    goto DONE;
  }
  if (check_objective_offset != objective_offset) {
    printf("Error: expected objective offset to be %f, but got %f\n", objective_offset, check_objective_offset);
    status = -1;
    goto DONE;
  }

  status = cuOptGetObjectiveCoefficients(problem, check_objective_coefficients);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective coefficients\n");
    goto DONE;
  }
  for (cuopt_int_t j = 0; j < num_variables; j++) {
    if (check_objective_coefficients[j] != objective_coefficients[j]) {
      printf("Error: expected objective coefficient %d to be %f, but got %f\n",
             j,
             objective_coefficients[j],
             check_objective_coefficients[j]);
      status = -1;
      goto DONE;
    }
  }

  status = cuOptGetConstraintMatrix(problem, check_row_offsets, check_column_indices, check_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting constraint matrix\n");
    goto DONE;
  }

  for (cuopt_int_t i = 0; i < num_constraints; i++) {
    if (check_row_offsets[i] != row_offsets[i]) {
      printf("Error: expected row offset %d to be %d, but got %d\n",
             i,
             row_offsets[i],
             check_row_offsets[i]);
      status = -1;
      goto DONE;
    }
  }

  for (cuopt_int_t k = 0; k < nnz; k++) {
    if (check_column_indices[k] != column_indices[k]) {
      printf("Error: expected column index %d to be %d, but got %d\n",
             k,
             column_indices[k],
             check_column_indices[k]);
      status = -1;
      goto DONE;
    }
  }

  for (cuopt_int_t k = 0; k < nnz; k++) {
    if (check_values[k] != values[k]) {
      printf("Error: expected value %d to be %f, but got %f\n", k, values[k], check_values[k]);
      status = -1;
      goto DONE;
    }
  }

  status = cuOptGetConstraintSense(problem, check_constraint_sense);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting constraint sense\n");
    goto DONE;
  }
  for (cuopt_int_t i = 0; i < num_constraints; i++) {
    if (check_constraint_sense[i] != constraint_sense[i]) {
      printf("Error: expected constraint sense %c to be %c, but got %c\n",
             i,
             constraint_sense[i],
             check_constraint_sense[i]);
      status = -1;
      goto DONE;
    }
  }

  status = cuOptGetConstraintRightHandSide(problem, check_rhs);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting constraint right hand side\n");
    goto DONE;
  }
  for (cuopt_int_t i = 0; i < num_constraints; i++) {
    if (check_rhs[i] != rhs[i]) {
      printf("Error: expected constraint right hand side %d to be %f, but got %f\n",
             i,
             rhs[i],
             check_rhs[i]);
      status = -1;
      goto DONE;
    }
  }

  status = cuOptGetVariableLowerBounds(problem, check_var_lower_bounds);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting variable lower bounds\n");
    goto DONE;
  }
  for (cuopt_int_t j = 0; j < num_variables; j++) {
    if (check_var_lower_bounds[j] != var_lower_bounds[j]) {
      printf("Error: expected variable lower bound %d to be %f, but got %f\n",
             j,
             var_lower_bounds[j],
             check_var_lower_bounds[j]);
      status = -1;
      goto DONE;
    }
  }

  status = cuOptGetVariableUpperBounds(problem, check_var_upper_bounds);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting variable upper bounds\n");
    goto DONE;
  }
  for (cuopt_int_t j = 0; j < num_variables; j++) {
    if (check_var_upper_bounds[j] != var_upper_bounds[j]) {
      printf("Error: expected variable upper bound %d to be %f, but got %f\n",
             j,
             var_upper_bounds[j],
             check_var_upper_bounds[j]);
      status = -1;
      goto DONE;
    }
  }

  status = cuOptGetVariableTypes(problem, check_variable_types);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting variable types\n");
    goto DONE;
  }
  for (cuopt_int_t j = 0; j < num_variables; j++) {
    if (check_variable_types[j] != variable_types[j]) {
      printf("Error: expected variable type %d to be %c, but got %c\n",
             j,
             variable_types[j],
             check_variable_types[j]);
      status = -1;
      goto DONE;
    }
  }

DONE:
  free(check_objective_coefficients);
  free(check_row_offsets);
  free(check_column_indices);
  free(check_values);
  free(check_constraint_sense);
  free(check_rhs);
  free(check_var_lower_bounds);
  free(check_var_upper_bounds);
  free(check_variable_types);

  return status;
}

cuopt_int_t test_infeasible_problem()
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings = NULL;
  cuOptSolution solution = NULL;


  /* Solve the following problem
  minimize 0
  subject to
              -0.5 X1 +  1.0 X2          >= .5     : row 1
               2.0 X1 -1.0 X2            >= 3.0    : row 2
               3.0 X1  + 1.0 X2          <= 6.0    : row 3
                         1.0 X5          <= 2.0    : row 4
               3.0 X4  -1.0 X5           <=  2.0   : row 5
                        1.0 X4           >= 5.0    : row 6
              1.0 X1 +  1.0 X5           <= 10.0   : row 7
              1.0 X1 +  2.0 X2 + 1.0 X4  <= 14.0   : row 8
              1.0 X2 +  1.0 X4           >= 1.0    : row 9

              X1, X2, X4, X5 >= 0
              0   1   2   3
 */

  cuopt_int_t num_variables = 4;
  cuopt_int_t num_constraints = 9;
  cuopt_int_t nnz = 17;
  cuopt_int_t row_offsets[] = {0, 2, 4, 6, 7, 9, 10, 12, 15, 17};
  // clang-format off
  //                               row1,      row2,     row3, row4,      row5,row6,      row7,          row8,       row9
  cuopt_int_t column_indices[] = {0,      1,   0,    1,   0,    1,   3,   2,   3,    2,    0,   3,    0,   1,  2,     1,   2};
  cuopt_float_t values[] =       {-0.5, 1.0, 2.0, -1.0, 3.0,  1.0, 1.0, 3.0, -1.0,  1.0, 1.0, 1.0,   1.0, 2.0, 1.0, 1.0, 1.0};
  // clang-format on
  cuopt_float_t rhs[] = {0.5, 3.0, 6.0, 2.0, 2.0, 5.0, 10.0, 14.0, 1.0};
  char constraint_sense[] = {CUOPT_GREATER_THAN, CUOPT_GREATER_THAN,
                            CUOPT_LESS_THAN, CUOPT_LESS_THAN, CUOPT_LESS_THAN,
                            CUOPT_GREATER_THAN, CUOPT_LESS_THAN, CUOPT_LESS_THAN, CUOPT_GREATER_THAN};
  cuopt_float_t var_lower_bounds[] = {0.0, 0.0, 0.0, 0.0};
  cuopt_float_t var_upper_bounds[] = {CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY};
  char variable_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
  cuopt_float_t objective_coefficients[] = {0.0, 0.0, 0.0, 0.0};

  cuopt_float_t time;
  cuopt_int_t termination_status;
  cuopt_float_t objective_value;

  cuopt_int_t status = cuOptCreateProblem(num_constraints,
                                      num_variables,
                                      CUOPT_MINIMIZE,
                                      0.0,
                                      objective_coefficients,
                                      row_offsets,
                                      column_indices,
                                      values,
                                      constraint_sense,
                                      rhs,
                                      var_lower_bounds,
                                      var_upper_bounds,
                                      variable_types,
                                      &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating problem\n");
    goto DONE;
  }

  status = check_problem(problem,
                         num_constraints,
                         num_variables,
                         nnz,
                         CUOPT_MINIMIZE,
                         0.0,
                         objective_coefficients,
                         row_offsets,
                         column_indices,
                         values,
                         constraint_sense,
                         rhs,
                         var_lower_bounds,
                         var_upper_bounds,
                         variable_types);
  if (status != CUOPT_SUCCESS) {
    printf("Error checking problem\n");
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings\n");
    goto DONE;
  };
  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_DUAL_SIMPLEX);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting parameter\n");
    goto DONE;
  }
  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem\n");
    goto DONE;
  }
  status = cuOptGetSolveTime(solution, &time);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solve time\n");
    goto DONE;
  }
  status = cuOptGetTerminationStatus(solution, &termination_status);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status\n");
    goto DONE;
  }
  if (termination_status != CUOPT_TERIMINATION_STATUS_INFEASIBLE) {
    printf("Error: expected termination status to be %d, but got %d\n",
           CUOPT_TERIMINATION_STATUS_INFEASIBLE,
           termination_status);
    status = -1;
    goto DONE;
  }
  status = cuOptGetObjectiveValue(solution, &objective_value);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value\n");
    goto DONE;
  }
  printf("Solve finished with termination status %s (%d) in %f seconds\n",
         termination_status_to_string(termination_status),
         termination_status,
         time);
  printf("Objective value: %f\n", objective_value);
DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}


cuopt_int_t test_ranged_problem(cuopt_int_t *termination_status_ptr, cuopt_float_t *objective_ptr)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings = NULL;
  cuOptSolution solution = NULL;

  // maximize obj: 5 * x + 8 * y;
  // subject to c1: 2*x + 3*y <= 12;
  // subject to c2: 3*x + y <= 6;
  // subject to c3: 2 <= x + 2*y <= 8;
  // subject to x_limit: 0 <= x <= 10;
  // subject to y_limit: 0 <= y <= 10;

  cuopt_int_t num_variables = 2;
  cuopt_int_t num_constraints = 3;
  cuopt_int_t nnz = 6;
  cuopt_int_t objective_sense = CUOPT_MAXIMIZE;
  cuopt_float_t objective_offset = 0.0;
  cuopt_float_t objective_coefficients[] = {5.0, 8.0};
  cuopt_int_t row_offsets[] = {0, 2, 4, 6};
  cuopt_int_t column_indices[] = {0, 1, 0, 1, 0, 1};
  cuopt_float_t values[] = {2.0, 3.0, 3.0, 1.0, 1.0, 2.0};
  cuopt_float_t constraint_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY, 2.0};
  cuopt_float_t constraint_upper_bounds[] = {12.0, 6.0, 8.0};
  cuopt_float_t constraint_lower_bounds_check[] = {1.0, 1.0, 1.0};
  cuopt_float_t constraint_upper_bounds_check[] = {1.0, 1.0, 1.0};
  cuopt_float_t variable_lower_bounds[] = {0.0, 0.0};
  cuopt_float_t variable_upper_bounds[] = {10.0, 10.0};
  char variable_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
  cuopt_int_t status;

  status = cuOptCreateRangedProblem(num_constraints,
                                    num_variables,
                                    objective_sense,
                                    objective_offset,
                                    objective_coefficients,
                                    row_offsets,
                                    column_indices,
                                    values,
                                    constraint_lower_bounds,
                                    constraint_upper_bounds,
                                    variable_lower_bounds,
                                    variable_upper_bounds,
                                    variable_types,
                                    &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating problem\n");
    goto DONE;
  }

  status = cuOptGetConstraintLowerBounds(problem, constraint_lower_bounds_check);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting constraint lower bounds\n");
    goto DONE;
  }

  status = cuOptGetConstraintUpperBounds(problem, constraint_upper_bounds_check);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting constraint upper bounds\n");
    goto DONE;
  }

  for (cuopt_int_t i = 0; i < num_constraints; i++) {
    if (constraint_lower_bounds_check[i] != constraint_lower_bounds[i]) {
      printf("Error: expected constraint lower bound %d to be %f, but got %f\n",
             i, constraint_lower_bounds[i], constraint_lower_bounds_check[i]);
      status = -1;
      goto DONE;
    }
    if (constraint_upper_bounds_check[i] != constraint_upper_bounds[i]) {
      printf("Error: expected constraint upper bound %d to be %f, but got %f\n",
             i, constraint_upper_bounds[i], constraint_upper_bounds_check[i]);
      status = -1;
      goto DONE;
    }
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings\n");
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_DUAL_SIMPLEX);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting parameter\n");
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem\n");
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, termination_status_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status\n");
    goto DONE;
  }

  status = cuOptGetObjectiveValue(solution, objective_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value\n");
    goto DONE;
  }

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}
