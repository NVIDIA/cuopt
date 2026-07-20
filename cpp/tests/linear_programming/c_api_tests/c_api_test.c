/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "c_api_tests.h"

#include <cuopt/mathematical_optimization/cuopt_c.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    case CUOPT_TERMINATION_STATUS_OPTIMAL:
      return "Optimal";
    case CUOPT_TERMINATION_STATUS_INFEASIBLE:
      return "Infeasible";
    case CUOPT_TERMINATION_STATUS_UNBOUNDED:
      return "Unbounded";
    case CUOPT_TERMINATION_STATUS_ITERATION_LIMIT:
      return "Iteration limit";
    case CUOPT_TERMINATION_STATUS_TIME_LIMIT:
      return "Time limit";
    case CUOPT_TERMINATION_STATUS_NUMERICAL_ERROR:
      return "Numerical error";
    case CUOPT_TERMINATION_STATUS_PRIMAL_FEASIBLE:
      return "Primal feasible";
    case CUOPT_TERMINATION_STATUS_FEASIBLE_FOUND:
      return "Feasible found";
  }
  return "Unknown";
}

int test_int_size() { return cuOptGetIntSize(); }

int test_float_size() { return cuOptGetFloatSize(); }

cuopt_int_t test_missing_file()
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
  cuopt_int_t status               = cuOptReadProblem("missing_file.mps", &problem);
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);
  return status;
}

cuopt_int_t test_bad_parameter_name()
{
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

typedef struct mip_callback_context_t {
  cuopt_int_t n_variables;
  int get_calls;
  int set_calls;
  int error;
  cuopt_float_t last_objective;
  cuopt_float_t last_solution_bound;
  cuopt_float_t* last_solution;
} mip_callback_context_t;

static void mip_get_solution_callback(const cuopt_float_t* solution,
                                      const cuopt_float_t* objective_value,
                                      const cuopt_float_t* solution_bound,
                                      void* user_data)
{
  mip_callback_context_t* context = (mip_callback_context_t*)user_data;
  if (context == NULL) { return; }
  context->get_calls += 1;
  if (context->last_solution == NULL) {
    context->last_solution = (cuopt_float_t*)malloc(context->n_variables * sizeof(cuopt_float_t));
    if (context->last_solution == NULL) {
      context->error = 1;
      return;
    }
  }
  memcpy(context->last_solution, solution, context->n_variables * sizeof(cuopt_float_t));
  memcpy(&context->last_objective, objective_value, sizeof(cuopt_float_t));
  memcpy(&context->last_solution_bound, solution_bound, sizeof(cuopt_float_t));
}

static void mip_set_solution_callback(cuopt_float_t* solution,
                                      cuopt_float_t* objective_value,
                                      const cuopt_float_t* solution_bound,
                                      void* user_data)
{
  mip_callback_context_t* context = (mip_callback_context_t*)user_data;
  if (context == NULL) { return; }
  context->set_calls += 1;
  memcpy(&context->last_solution_bound, solution_bound, sizeof(cuopt_float_t));
  if (context->last_solution == NULL) { return; }
  memcpy(solution, context->last_solution, context->n_variables * sizeof(cuopt_float_t));
  memcpy(objective_value, &context->last_objective, sizeof(cuopt_float_t));
}

static cuopt_int_t test_mip_callbacks_internal(int include_set_callback)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
  mip_callback_context_t context   = {0};

#define NUM_ITEMS       8
#define NUM_CONSTRAINTS 1
  cuopt_int_t num_items    = NUM_ITEMS;
  cuopt_float_t max_weight = 102;
  cuopt_float_t value[]    = {15, 100, 90, 60, 40, 15, 10, 1};
  cuopt_float_t weight[]   = {2, 20, 20, 30, 40, 30, 60, 10};

  cuopt_int_t num_variables   = NUM_ITEMS;
  cuopt_int_t num_constraints = NUM_CONSTRAINTS;

  cuopt_int_t row_offsets[] = {0, NUM_ITEMS};
  cuopt_int_t column_indices[NUM_ITEMS];

  cuopt_float_t rhs[]     = {max_weight};
  char constraint_sense[] = {CUOPT_LESS_THAN};
  cuopt_float_t lower_bounds[NUM_ITEMS];
  cuopt_float_t upper_bounds[NUM_ITEMS];
  char variable_types[NUM_ITEMS];
  cuopt_int_t status;

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
                              CUOPT_MAXIMIZE,
                              0,
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

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings\n");
    goto DONE;
  }

  context.n_variables = num_variables;
  status = cuOptSetMIPGetSolutionCallback(settings, mip_get_solution_callback, &context);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting get-solution callback\n");
    goto DONE;
  }

  if (include_set_callback) {
    status = cuOptSetMIPSetSolutionCallback(settings, mip_set_solution_callback, &context);
    if (status != CUOPT_SUCCESS) {
      printf("Error setting set-solution callback\n");
      goto DONE;
    }
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem\n");
    goto DONE;
  }

  if (context.error != 0) {
    printf("Error in callback data transfer\n");
    status = CUOPT_INVALID_ARGUMENT;
    goto DONE;
  }

  if (context.last_solution_bound != context.last_solution_bound) {
    printf("Error reading solution bound in callback\n");
    status = CUOPT_INVALID_ARGUMENT;
    goto DONE;
  }

  if (context.get_calls < 1) {
    printf("Expected get-solution callback to be called at least once\n");
    status = CUOPT_INVALID_ARGUMENT;
    goto DONE;
  }
  if (include_set_callback && context.set_calls < 1) {
    printf("Expected set-solution callback to be called at least once\n");
    status = CUOPT_INVALID_ARGUMENT;
    goto DONE;
  }

DONE:
  if (context.last_solution != NULL) { free(context.last_solution); }
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);
  return status;
}

cuopt_int_t test_mip_get_callbacks_only() { return test_mip_callbacks_internal(0); }

cuopt_int_t test_mip_get_set_callbacks() { return test_mip_callbacks_internal(1); }

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

  cuopt_float_t rhs[]     = {max_weight};
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
  if (termination_status != CUOPT_TERMINATION_STATUS_OPTIMAL) {
    printf("Error: expected termination status to be %d, but got %d\n",
           CUOPT_TERMINATION_STATUS_OPTIMAL,
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

int solve_mps_file(const char* filename,
                   double time_limit,
                   double iteration_limit,
                   int* termination_status_ptr,
                   double* solve_time_ptr,
                   int method)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
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
    cuopt_int_t error_string_status =
      cuOptGetErrorString(solution, error_string, ERROR_BUFFER_SIZE);
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
  check_row_offsets            = (cuopt_int_t*)malloc((num_constraints + 1) * sizeof(cuopt_int_t));
  check_column_indices         = (cuopt_int_t*)malloc(nnz * sizeof(cuopt_int_t));
  check_values                 = (cuopt_float_t*)malloc(nnz * sizeof(cuopt_float_t));
  check_constraint_sense       = (char*)malloc(num_constraints * sizeof(char));
  check_rhs                    = (cuopt_float_t*)malloc(num_constraints * sizeof(cuopt_float_t));
  check_var_lower_bounds       = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  check_var_upper_bounds       = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  check_variable_types         = (char*)malloc(num_variables * sizeof(char));

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
    printf("Error: expected objective offset to be %f, but got %f\n",
           objective_offset,
           check_objective_offset);
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
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

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

  cuopt_int_t num_variables   = 4;
  cuopt_int_t num_constraints = 9;
  cuopt_int_t nnz             = 17;
  cuopt_int_t row_offsets[]   = {0, 2, 4, 6, 7, 9, 10, 12, 15, 17};
  // clang-format off
  //                               row1,      row2,     row3, row4,      row5,row6,      row7,          row8,       row9
  cuopt_int_t column_indices[] = {0,      1,   0,    1,   0,    1,   3,   2,   3,    2,    0,   3,    0,   1,  2,     1,   2};
  cuopt_float_t values[] =       {-0.5, 1.0, 2.0, -1.0, 3.0,  1.0, 1.0, 3.0, -1.0,  1.0, 1.0, 1.0,   1.0, 2.0, 1.0, 1.0, 1.0};
  // clang-format on
  cuopt_float_t rhs[]              = {0.5, 3.0, 6.0, 2.0, 2.0, 5.0, 10.0, 14.0, 1.0};
  char constraint_sense[]          = {CUOPT_GREATER_THAN,
                                      CUOPT_GREATER_THAN,
                                      CUOPT_LESS_THAN,
                                      CUOPT_LESS_THAN,
                                      CUOPT_LESS_THAN,
                                      CUOPT_GREATER_THAN,
                                      CUOPT_LESS_THAN,
                                      CUOPT_LESS_THAN,
                                      CUOPT_GREATER_THAN};
  cuopt_float_t var_lower_bounds[] = {0.0, 0.0, 0.0, 0.0};
  cuopt_float_t var_upper_bounds[] = {
    CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY};
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
  if (termination_status != CUOPT_TERMINATION_STATUS_INFEASIBLE) {
    printf("Error: expected termination status to be %d, but got %d\n",
           CUOPT_TERMINATION_STATUS_INFEASIBLE,
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

cuopt_int_t test_ranged_problem(cuopt_int_t* termination_status_ptr, cuopt_float_t* objective_ptr)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  // maximize obj: 5 * x + 8 * y;
  // subject to c1: 2*x + 3*y <= 12;
  // subject to c2: 3*x + y <= 6;
  // subject to c3: 2 <= x + 2*y <= 8;
  // subject to x_limit: 0 <= x <= 10;
  // subject to y_limit: 0 <= y <= 10;

  cuopt_int_t num_variables                     = 2;
  cuopt_int_t num_constraints                   = 3;
  cuopt_int_t nnz                               = 6;
  cuopt_int_t objective_sense                   = CUOPT_MAXIMIZE;
  cuopt_float_t objective_offset                = 0.0;
  cuopt_float_t objective_coefficients[]        = {5.0, 8.0};
  cuopt_int_t row_offsets[]                     = {0, 2, 4, 6};
  cuopt_int_t column_indices[]                  = {0, 1, 0, 1, 0, 1};
  cuopt_float_t values[]                        = {2.0, 3.0, 3.0, 1.0, 1.0, 2.0};
  cuopt_float_t constraint_lower_bounds[]       = {-CUOPT_INFINITY, -CUOPT_INFINITY, 2.0};
  cuopt_float_t constraint_upper_bounds[]       = {12.0, 6.0, 8.0};
  cuopt_float_t constraint_lower_bounds_check[] = {1.0, 1.0, 1.0};
  cuopt_float_t constraint_upper_bounds_check[] = {1.0, 1.0, 1.0};
  cuopt_float_t variable_lower_bounds[]         = {0.0, 0.0};
  cuopt_float_t variable_upper_bounds[]         = {10.0, 10.0};
  char variable_types[]                         = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
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
             i,
             constraint_lower_bounds[i],
             constraint_lower_bounds_check[i]);
      status = -1;
      goto DONE;
    }
    if (constraint_upper_bounds_check[i] != constraint_upper_bounds[i]) {
      printf("Error: expected constraint upper bound %d to be %f, but got %f\n",
             i,
             constraint_upper_bounds[i],
             constraint_upper_bounds_check[i]);
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

cuopt_int_t test_semi_continuous_problem(cuopt_int_t* termination_status_ptr,
                                         cuopt_float_t* objective_ptr,
                                         cuopt_float_t* solution_values)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  /* Minimize x with x semi-continuous and x + y = 1.
     Since x is either 0 or in [5, 10], the optimum is x = 0, y = 1.
     If CUOPT_SEMI_CONTINUOUS is treated as integer, this model is infeasible. */
  cuopt_int_t num_variables              = 2;
  cuopt_int_t num_constraints            = 1;
  cuopt_int_t row_offsets[]              = {0, 2};
  cuopt_int_t column_indices[]           = {0, 1};
  cuopt_float_t values[]                 = {1.0, 1.0};
  char constraint_sense[]                = {CUOPT_EQUAL};
  cuopt_float_t rhs[]                    = {1.0};
  cuopt_float_t lower_bounds[]           = {5.0, 0.0};
  cuopt_float_t upper_bounds[]           = {10.0, 1.0};
  char variable_types[]                  = {CUOPT_SEMI_CONTINUOUS, CUOPT_CONTINUOUS};
  cuopt_float_t objective_coefficients[] = {1.0, 0.0};
  char check_variable_types[2];
  cuopt_int_t is_mip;
  cuopt_int_t status;

  status = cuOptCreateProblem(num_constraints,
                              num_variables,
                              CUOPT_MINIMIZE,
                              0.0,
                              objective_coefficients,
                              row_offsets,
                              column_indices,
                              values,
                              constraint_sense,
                              rhs,
                              lower_bounds,
                              upper_bounds,
                              variable_types,
                              &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating semi-continuous problem\n");
    goto DONE;
  }

  status = cuOptGetVariableTypes(problem, check_variable_types);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting variable types for semi-continuous problem\n");
    goto DONE;
  }
  if (check_variable_types[0] != CUOPT_SEMI_CONTINUOUS ||
      check_variable_types[1] != CUOPT_CONTINUOUS) {
    printf("Error: semi-continuous variable types were not preserved\n");
    status = -1;
    goto DONE;
  }

  status = cuOptIsMIP(problem, &is_mip);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting MIP flag for semi-continuous problem\n");
    goto DONE;
  }
  if (!is_mip) {
    printf("Error: semi-continuous problem was not detected as a MIP\n");
    status = -1;
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings\n");
    goto DONE;
  }
  status = cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, 10.0);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting time limit\n");
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving semi-continuous problem\n");
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

  status = cuOptGetPrimalSolution(solution, solution_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting primal solution\n");
    goto DONE;
  }

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

// Test invalid bounds scenario (what MOI wrapper was producing)
cuopt_int_t test_invalid_bounds(cuopt_int_t test_mip)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  /* Test the invalid bounds scenario:
     maximize 2*x
     subject to:
     x >= 0.2
     x <= 0.5
     x is binary (0 or 1)

     After MOI wrapper processing:
     - Lower bound = ceil(max(0.0, 0.2)) = 1.0
     - Upper bound = floor(min(1.0, 0.5)) = 0.0
     - Result: 1.0 <= x <= 0.0 (INVALID!)
  */

  cuopt_int_t num_variables   = 1;
  cuopt_int_t num_constraints = 2;
  cuopt_int_t nnz             = 2;

  // CSR format constraint matrix
  // From the constraints:
  // x >= 0.2
  // x <= 0.5
  cuopt_int_t row_offsets[]    = {0, 1, 2};
  cuopt_int_t column_indices[] = {0, 0};
  cuopt_float_t values[]       = {1.0, 1.0};

  // Objective coefficients
  // From the objective function: maximize 2*x
  cuopt_float_t objective_coefficients[] = {2.0};

  // Constraint bounds
  // From the constraints:
  // x >= 0.2
  // x <= 0.5
  cuopt_float_t constraint_upper_bounds[] = {CUOPT_INFINITY, 0.5};
  cuopt_float_t constraint_lower_bounds[] = {0.2, -CUOPT_INFINITY};

  // Variable bounds - INVALID: lower > upper
  // After MOI wrapper processing:
  cuopt_float_t var_lower_bounds[] = {1.0};  // ceil(max(0.0, 0.2)) = 1.0
  cuopt_float_t var_upper_bounds[] = {0.0};  // floor(min(1.0, 0.5)) = 0.0

  // Variable types (binary)
  char variable_types[] = {CUOPT_INTEGER};  // Binary variable
  if (!test_mip) variable_types[0] = CUOPT_CONTINUOUS;

  cuopt_int_t status;
  cuopt_float_t time;
  cuopt_int_t termination_status;
  cuopt_float_t objective_value;

  printf("Testing invalid bounds scenario (MOI wrapper issue)...\n");
  printf("Problem: Binary variable with bounds 1.0 <= x <= 0.0 (INVALID!)\n");

  // Create the problem
  status = cuOptCreateRangedProblem(num_constraints,
                                    num_variables,
                                    CUOPT_MAXIMIZE,  // maximize
                                    0.0,             // objective offset
                                    objective_coefficients,
                                    row_offsets,
                                    column_indices,
                                    values,
                                    constraint_lower_bounds,
                                    constraint_upper_bounds,
                                    var_lower_bounds,
                                    var_upper_bounds,
                                    variable_types,
                                    &problem);

  printf("cuOptCreateRangedProblem returned: %d\n", status);

  if (status != CUOPT_SUCCESS) {
    printf("✗ Unexpected error: %d\n", status);
    goto DONE;
  }

  // If we get here, the problem was created successfully
  printf("✓ Problem created successfully\n");

  // Create solver settings
  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  // Solve the problem
  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem: %d\n", status);
    goto DONE;
  }

  // Get solution information
  status = cuOptGetSolveTime(solution, &time);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solve time: %d\n", status);
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, &termination_status);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status: %d\n", status);
    goto DONE;
  }
  if (termination_status != CUOPT_TERMINATION_STATUS_INFEASIBLE) {
    printf("Error: expected termination status to be %d, but got %d\n",
           CUOPT_TERMINATION_STATUS_INFEASIBLE,
           termination_status);
    status = CUOPT_VALIDATION_ERROR;
    goto DONE;
  } else {
    printf("✓ Problem found infeasible as expected\n");
    status = CUOPT_SUCCESS;
    goto DONE;
  }

  status = cuOptGetObjectiveValue(solution, &objective_value);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

  // Print results
  printf("\nResults:\n");
  printf("--------\n");
  printf("Termination status: %s (%d)\n",
         termination_status_to_string(termination_status),
         termination_status);
  printf("Solve time: %f seconds\n", time);
  printf("Objective value: %f\n", objective_value);

  // Get and print solution variables
  cuopt_float_t* solution_values = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  status                         = cuOptGetPrimalSolution(solution, solution_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solution values: %d\n", status);
    free(solution_values);
    goto DONE;
  }

  printf("\nSolution: \n");
  for (cuopt_int_t i = 0; i < num_variables; i++) {
    printf("x%d = %f\n", i + 1, solution_values[i]);
  }
  free(solution_values);

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

cuopt_int_t test_quadratic_problem(cuopt_int_t* termination_status_ptr,
                                   cuopt_float_t* objective_ptr)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  // minimize x1^2 + 4*x2^2 - 8*x1 - 16*x2
  // subject to x1 + x2 >= 5
  //         x1 >= 3
  //         x2 >= 0
  //         x1 <= 10
  //         x2 <= 10

  cuopt_int_t num_variables              = 2;
  cuopt_int_t num_constraints            = 1;
  cuopt_int_t objective_sense            = CUOPT_MINIMIZE;
  cuopt_float_t objective_offset         = 0.0;
  cuopt_float_t objective_coefficients[] = {-8.0, -16.0};

  cuopt_int_t Q_row_index[] = {0, 1};
  cuopt_int_t Q_col_index[] = {0, 1};
  cuopt_float_t Q_coeff[]   = {1.0, 4.0};

  cuopt_int_t row_offsets[]    = {0, 2};
  cuopt_int_t column_indices[] = {0, 1};
  cuopt_float_t values[]       = {1.0, 1.0};

  cuopt_float_t constraint_bounds[] = {5.0};
  char constraint_sense[]           = {'G'};

  cuopt_float_t var_lower_bounds[] = {3.0, 0.0};
  cuopt_float_t var_upper_bounds[] = {10.0, 10.0};
  char variable_types[]            = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  cuopt_int_t status;

  status = cuOptCreateProblem(num_constraints,
                              num_variables,
                              objective_sense,
                              objective_offset,
                              objective_coefficients,
                              row_offsets,
                              column_indices,
                              values,
                              constraint_sense,
                              constraint_bounds,
                              var_lower_bounds,
                              var_upper_bounds,
                              variable_types,
                              &problem);

  if (status != CUOPT_SUCCESS) {
    printf("Error creating problem: %d\n", status);
    goto DONE;
  }

  status = cuOptSetQuadraticObjective(problem, 2, Q_row_index, Q_col_index, Q_coeff);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting quadratic objective: %d\n", status);
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem: %d\n", status);
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, termination_status_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status: %d\n", status);
    goto DONE;
  }

  status = cuOptGetObjectiveValue(solution, objective_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

cuopt_int_t test_quadratic_ranged_problem(cuopt_int_t* termination_status_ptr,
                                          cuopt_float_t* objective_ptr)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  // minimize x1^2 + 4*x2^2 - 8*x1 - 16*x2
  // subject to x1 + x2 >= 5
  //         x1 >= 3
  //         x2 >= 0
  //         x1 <= 10
  //         x2 <= 10
  cuopt_int_t num_variables              = 2;
  cuopt_int_t num_constraints            = 1;
  cuopt_int_t objective_sense            = CUOPT_MINIMIZE;
  cuopt_float_t objective_offset         = 0.0;
  cuopt_float_t objective_coefficients[] = {-8.0, -16.0};

  cuopt_int_t Q_row_index[] = {0, 1};
  cuopt_int_t Q_col_index[] = {0, 1};
  cuopt_float_t Q_coeff[]   = {1.0, 4.0};

  cuopt_int_t row_offsets[]    = {0, 2};
  cuopt_int_t column_indices[] = {0, 1};
  cuopt_float_t values[]       = {1.0, 1.0};

  cuopt_float_t constraint_lower_bounds[] = {5.0};
  cuopt_float_t constraint_upper_bounds[] = {100.0};

  cuopt_float_t var_lower_bounds[] = {3.0, 0.0};
  cuopt_float_t var_upper_bounds[] = {10.0, 10.0};

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
                                    var_lower_bounds,
                                    var_upper_bounds,
                                    NULL,
                                    &problem);

  if (status != CUOPT_SUCCESS) {
    printf("Error creating problem: %d\n", status);
    goto DONE;
  }

  status = cuOptSetQuadraticObjective(problem, 2, Q_row_index, Q_col_index, Q_coeff);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting quadratic objective: %d\n", status);
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem: %d\n", status);
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, termination_status_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status: %d\n", status);
    goto DONE;
  }

  status = cuOptGetObjectiveValue(solution, objective_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

cuopt_int_t test_quadratic_constraint_problem(cuopt_int_t* termination_status_ptr,
                                              cuopt_float_t* objective_ptr,
                                              cuopt_float_t* solution_values)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  // Same QCQP as python/cuopt/cuopt/tests/socp/test_socp.py::build_socp_1:
  //   min  3*x0 + 2*x1 + x2
  //   s.t. x0^2 + x1^2 + x2^2 - y^2 <= 0
  //        x0 + x1 + 3*x2 >= 1
  //        0 <= y <= 5
  //   (x0, x1, x2 free)
  cuopt_int_t num_variables         = 4;
  cuopt_int_t num_linear_constraints = 1;
  cuopt_int_t objective_sense            = CUOPT_MINIMIZE;
  cuopt_float_t objective_offset         = 0.0;
  cuopt_float_t objective_coefficients[] = {3.0, 2.0, 1.0, 0.0};

  cuopt_int_t row_offsets[]    = {0, 3};
  cuopt_int_t column_indices[] = {0, 1, 2};
  cuopt_float_t values[]       = {1.0, 1.0, 3.0};

  cuopt_float_t constraint_bounds[] = {1.0};
  char constraint_sense[]           = {CUOPT_GREATER_THAN};

  cuopt_float_t var_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY, -CUOPT_INFINITY, 0.0};
  cuopt_float_t var_upper_bounds[] = {
    CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY, 5.0};
  char variable_types[]            = {CUOPT_CONTINUOUS,
                           CUOPT_CONTINUOUS,
                           CUOPT_CONTINUOUS,
                           CUOPT_CONTINUOUS};

  cuopt_int_t qc_row_index[] = {0, 1, 2, 3};
  cuopt_int_t qc_col_index[] = {0, 1, 2, 3};
  cuopt_float_t qc_coeff[]   = {1.0, 1.0, 1.0, -1.0};
  // Canonical COO: diagonal tail +s and head -s (one entry per variable pair).

  cuopt_int_t status;

  status = cuOptCreateProblem(num_linear_constraints,
                              num_variables,
                              objective_sense,
                              objective_offset,
                              objective_coefficients,
                              row_offsets,
                              column_indices,
                              values,
                              constraint_sense,
                              constraint_bounds,
                              var_lower_bounds,
                              var_upper_bounds,
                              variable_types,
                              &problem);

  if (status != CUOPT_SUCCESS) {
    printf("Error creating problem: %d\n", status);
    goto DONE;
  }

  status = cuOptAddQuadraticConstraint(problem,
                                       4,
                                       qc_row_index,
                                       qc_col_index,
                                       qc_coeff,
                                       0,
                                       NULL,
                                       NULL,
                                       CUOPT_LESS_THAN,
                                       0.0);
  if (status != CUOPT_SUCCESS) {
    printf("Error adding quadratic constraint: %d\n", status);
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_BARRIER);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting barrier method: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem: %d\n", status);
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, termination_status_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status: %d\n", status);
    goto DONE;
  }

  status = cuOptGetObjectiveValue(solution, objective_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

  status = cuOptGetPrimalSolution(solution, solution_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting primal solution: %d\n", status);
    goto DONE;
  }

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

cuopt_int_t test_general_quadratic_constraint_problem(cuopt_int_t* termination_status_ptr,
                                                      cuopt_float_t* objective_ptr,
                                                      cuopt_float_t* solution_values)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  // minimize x0 + x1
  // subject to 2*x0^2 + 3*x0*x1 + 2*x1^2 <= 1  (unsymmetric Q, general convex)
  //            x0 - x1 = 0
  // Q is given with only upper triangle entry for the cross term:
  //   (0,0,2), (0,1,3), (1,1,2)
  // Hessian H = (Q + Q^T)/2 is [4 3; 3 4], eigenvalues 1 and 7 (PD).
  // With x0 = x1: quadratic = 7*x0^2 <= 1, min 2*x0 at x0 = -1/sqrt(7)
  // Optimal objective = -2/sqrt(7) ≈ -0.755929
  cuopt_int_t num_variables          = 2;
  cuopt_int_t num_linear_constraints = 1;
  cuopt_int_t objective_sense        = CUOPT_MINIMIZE;
  cuopt_float_t objective_offset     = 0.0;
  cuopt_float_t objective_coefficients[] = {1.0, 1.0};

  cuopt_int_t row_offsets[]    = {0, 2};
  cuopt_int_t column_indices[] = {0, 1};
  cuopt_float_t values[]       = {1.0, -1.0};

  cuopt_float_t constraint_bounds[] = {0.0};
  char constraint_sense[]           = {CUOPT_EQUAL};

  cuopt_float_t var_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY};
  cuopt_float_t var_upper_bounds[] = {CUOPT_INFINITY, CUOPT_INFINITY};
  char variable_types[]            = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  // Canonical COO: (0,0,2), (0,1,3), (1,1,2) — one cross coefficient on (0,1).
  cuopt_int_t qc_row_index[] = {0, 0, 1};
  cuopt_int_t qc_col_index[] = {0, 1, 1};
  cuopt_float_t qc_coeff[]   = {2.0, 3.0, 2.0};

  cuopt_int_t status;

  status = cuOptCreateProblem(num_linear_constraints,
                              num_variables,
                              objective_sense,
                              objective_offset,
                              objective_coefficients,
                              row_offsets,
                              column_indices,
                              values,
                              constraint_sense,
                              constraint_bounds,
                              var_lower_bounds,
                              var_upper_bounds,
                              variable_types,
                              &problem);

  if (status != CUOPT_SUCCESS) {
    printf("Error creating problem: %d\n", status);
    goto DONE;
  }

  status = cuOptAddQuadraticConstraint(problem,
                                       3,
                                       qc_row_index,
                                       qc_col_index,
                                       qc_coeff,
                                       0,
                                       NULL,
                                       NULL,
                                       CUOPT_LESS_THAN,
                                       1.0);
  if (status != CUOPT_SUCCESS) {
    printf("Error adding quadratic constraint: %d\n", status);
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_BARRIER);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting barrier method: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem: %d\n", status);
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, termination_status_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status: %d\n", status);
    goto DONE;
  }

  status = cuOptGetObjectiveValue(solution, objective_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

  status = cuOptGetPrimalSolution(solution, solution_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting primal solution: %d\n", status);
    goto DONE;
  }

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

cuopt_int_t test_rotated_soc_constraint_problem(cuopt_int_t* termination_status_ptr,
                                                cuopt_float_t* objective_ptr,
                                                cuopt_float_t* solution_values)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  // Same as docs/cuopt-c/convex/examples/rotated_socp_example.c:
  //   min  x3 + x4
  //   s.t. x1 + x2 >= 2
  //        x1^2 + x2^2 - x3*x4 <= 0   (||tail||^2 <= x3*x4)
  //        x3 >= 0, x4 >= 0
  cuopt_int_t num_variables          = 4;
  cuopt_int_t num_linear_constraints = 1;
  cuopt_int_t objective_sense        = CUOPT_MINIMIZE;
  cuopt_float_t objective_offset     = 0.0;
  cuopt_float_t objective_coefficients[] = {0.0, 0.0, 1.0, 1.0};

  cuopt_int_t row_offsets[]    = {0, 2};
  cuopt_int_t column_indices[] = {0, 1};
  cuopt_float_t values[]       = {1.0, 1.0};

  cuopt_float_t constraint_bounds[] = {2.0};
  char constraint_sense[]           = {CUOPT_GREATER_THAN};

  cuopt_float_t var_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY, 0.0, 0.0};
  cuopt_float_t var_upper_bounds[] = {
    CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY};
  char variable_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  cuopt_int_t qc_row_index[] = {0, 1, 2};
  cuopt_int_t qc_col_index[] = {0, 1, 3};
  cuopt_float_t qc_coeff[]   = {1.0, 1.0, -1.0};

  cuopt_int_t status;

  status = cuOptCreateProblem(num_linear_constraints,
                              num_variables,
                              objective_sense,
                              objective_offset,
                              objective_coefficients,
                              row_offsets,
                              column_indices,
                              values,
                              constraint_sense,
                              constraint_bounds,
                              var_lower_bounds,
                              var_upper_bounds,
                              variable_types,
                              &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating problem: %d\n", status);
    goto DONE;
  }

  status = cuOptAddQuadraticConstraint(problem,
                                       3,
                                       qc_row_index,
                                       qc_col_index,
                                       qc_coeff,
                                       0,
                                       NULL,
                                       NULL,
                                       CUOPT_LESS_THAN,
                                       0.0);
  if (status != CUOPT_SUCCESS) {
    printf("Error adding quadratic constraint: %d\n", status);
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_BARRIER);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting barrier method: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem: %d\n", status);
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, termination_status_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status: %d\n", status);
    goto DONE;
  }

  status = cuOptGetObjectiveValue(solution, objective_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

  status = cuOptGetPrimalSolution(solution, solution_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting primal solution: %d\n", status);
    goto DONE;
  }

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

cuopt_int_t test_rotated_soc_standard_cross_term_problem(cuopt_int_t* termination_status_ptr,
                                                         cuopt_float_t* objective_ptr,
                                                         cuopt_float_t* solution_values)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  // Standard Lorentz rotated cone ||tail||^2 <= 2*x3*x4 written as
  // x1^2 + x2^2 - 2*x3*x4 <= 0 with canonical cross Q[x3, x4] = -2.
  //   min  x3 + x4
  //   s.t. x1 + x2 >= 2, x3 >= 0, x4 >= 0
  cuopt_int_t num_variables          = 4;
  cuopt_int_t num_linear_constraints = 1;
  cuopt_int_t objective_sense        = CUOPT_MINIMIZE;
  cuopt_float_t objective_offset     = 0.0;
  cuopt_float_t objective_coefficients[] = {0.0, 0.0, 1.0, 1.0};

  cuopt_int_t row_offsets[]    = {0, 2};
  cuopt_int_t column_indices[] = {0, 1};
  cuopt_float_t values[]       = {1.0, 1.0};

  cuopt_float_t constraint_bounds[] = {2.0};
  char constraint_sense[]           = {CUOPT_GREATER_THAN};

  cuopt_float_t var_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY, 0.0, 0.0};
  cuopt_float_t var_upper_bounds[] = {
    CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY};
  char variable_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  cuopt_int_t qc_row_index[] = {0, 1, 2};
  cuopt_int_t qc_col_index[] = {0, 1, 3};
  cuopt_float_t qc_coeff[]   = {1.0, 1.0, -2.0};

  cuopt_int_t status;

  status = cuOptCreateProblem(num_linear_constraints,
                              num_variables,
                              objective_sense,
                              objective_offset,
                              objective_coefficients,
                              row_offsets,
                              column_indices,
                              values,
                              constraint_sense,
                              constraint_bounds,
                              var_lower_bounds,
                              var_upper_bounds,
                              variable_types,
                              &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating problem: %d\n", status);
    goto DONE;
  }

  status = cuOptAddQuadraticConstraint(problem,
                                       3,
                                       qc_row_index,
                                       qc_col_index,
                                       qc_coeff,
                                       0,
                                       NULL,
                                       NULL,
                                       CUOPT_LESS_THAN,
                                       0.0);
  if (status != CUOPT_SUCCESS) {
    printf("Error adding quadratic constraint: %d\n", status);
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_BARRIER);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting barrier method: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem: %d\n", status);
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, termination_status_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status: %d\n", status);
    goto DONE;
  }

  status = cuOptGetObjectiveValue(solution, objective_ptr);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

  status = cuOptGetPrimalSolution(solution, solution_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting primal solution: %d\n", status);
    goto DONE;
  }

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

cuopt_int_t test_write_problem(const char* input_filename, const char* output_filename)
{
  cuOptOptimizationProblem problem      = NULL;
  cuOptOptimizationProblem problem_read = NULL;
  cuOptSolverSettings settings          = NULL;
  cuOptSolution solution                = NULL;
  cuopt_int_t status;
  cuopt_int_t termination_status;
  cuopt_float_t objective_value;

  /* Read the input problem */
  status = cuOptReadProblem(input_filename, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error reading problem from %s: %d\n", input_filename, status);
    goto DONE;
  }

  /* Write the problem to MPS file */
  status = cuOptWriteProblem(problem, output_filename, CUOPT_FILE_FORMAT_MPS);
  if (status != CUOPT_SUCCESS) {
    printf("Error writing problem to MPS: %d\n", status);
    goto DONE;
  }
  printf("Problem written to %s\n", output_filename);

  /* Read the problem back */
  status = cuOptReadProblem(output_filename, &problem_read);
  if (status != CUOPT_SUCCESS) {
    printf("Error reading problem from MPS: %d\n", status);
    goto DONE;
  }
  printf("Problem read back from %s\n", output_filename);

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_PDLP);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting method: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem_read, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem: %d\n", status);
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, &termination_status);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status: %d\n", status);
    goto DONE;
  }

  status = cuOptGetObjectiveValue(solution, &objective_value);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

  printf("Termination status: %d, Objective: %f\n", termination_status, objective_value);

  if (termination_status != CUOPT_TERMINATION_STATUS_OPTIMAL) {
    printf("Expected optimal status\n");
    status = -1;
    goto DONE;
  }

  printf("Write problem test passed\n");

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroyProblem(&problem_read);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);
  return status;
}

cuopt_int_t test_maximize_problem_dual_variables(cuopt_int_t method,
                                                 cuopt_int_t* termination_status_ptr,
                                                 cuopt_float_t* objective_ptr,
                                                 cuopt_float_t* dual_variables,
                                                 cuopt_float_t* reduced_costs,
                                                 cuopt_float_t* dual_obj_ptr)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  /* Solve the following problem
   maximize 4*x1 + x2 + 5*x3 + 3*x4
   subject to x1 - x2 - x3 + 3*x4 <= 1
              5*x1 + x2 + 3*x3 + 8*x4 <= 55
             -x1 + 2*x2 + 3*x3 -5*x4 <= 3
             x1, x2, x3, x4 >= 0
  */

  cuopt_int_t num_variables    = 4;
  cuopt_int_t num_constraints  = 3;
  cuopt_int_t nnz              = 12;
  cuopt_int_t row_offsets[]    = {0, 4, 8, 12};
  cuopt_int_t column_indices[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  cuopt_float_t values[]       = {1.0, -1.0, -1.0, 3.0, 5.0, 1.0, 3.0, 8.0, -1.0, 2.0, 3.0, -5.0};
  cuopt_float_t rhs[]          = {1.0, 55.0, 3.0};
  char constraint_sense[]      = {CUOPT_LESS_THAN, CUOPT_LESS_THAN, CUOPT_LESS_THAN};
  cuopt_float_t var_lower_bounds[] = {0.0, 0.0, 0.0, 0.0};
  cuopt_float_t var_upper_bounds[] = {
    CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY};
  char variable_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};
  cuopt_float_t objective_coefficients[] = {4.0, 1.0, 5.0, 3.0};

  cuopt_float_t time;
  cuopt_int_t i, j;

  cuopt_int_t status = cuOptCreateProblem(num_constraints,
                                          num_variables,
                                          CUOPT_MAXIMIZE,
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
                         CUOPT_MAXIMIZE,
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

  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, method);
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
  printf("Solve finished with termination status %s (%d) in %f seconds\n",
         termination_status_to_string(*termination_status_ptr),
         *termination_status_ptr,
         time);
  printf("Objective value: %f\n", *objective_ptr);

  /* Get and print solution variables */
  cuopt_float_t* solution_values = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  status                         = cuOptGetPrimalSolution(solution, solution_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solution values: %d\n", status);
    free(solution_values);
    goto DONE;
  }

  printf("\nSolution: \n");
  for (j = 0; j < num_variables; j++) {
    printf("x%d = %f\n", j + 1, solution_values[j]);
  }
  free(solution_values);

  status = cuOptGetDualSolution(solution, dual_variables);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting dual solution\n");
    goto DONE;
  }
  for (i = 0; i < num_constraints; i++) {
    printf("y%d = %f\n", i + 1, dual_variables[i]);
  }
  status = cuOptGetReducedCosts(solution, reduced_costs);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting reduced costs\n");
    goto DONE;
  }
  for (j = 0; j < num_variables; j++) {
    printf("z%d = %f\n", j + 1, reduced_costs[j]);
  }

  *dual_obj_ptr = 0.0;
  for (i = 0; i < num_constraints; i++) {
    *dual_obj_ptr += dual_variables[i] * rhs[i];
  }
DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

cuopt_int_t test_deterministic_bb(const char* filename,
                                  cuopt_int_t num_runs,
                                  cuopt_int_t num_threads,
                                  cuopt_float_t time_limit,
                                  cuopt_float_t work_limit)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuopt_float_t first_objective    = 0.0;
  cuopt_int_t first_status         = -1;
  cuopt_int_t status;
  cuopt_int_t run;

  printf(
    "Testing deterministic B&B: %s with %d threads, %d runs\n", filename, num_threads, num_runs);

  status = cuOptReadProblem(filename, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error reading problem: %d\n", status);
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_MIP_DETERMINISM_MODE, CUOPT_MODE_DETERMINISTIC);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting determinism mode: %d\n", status);
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_NUM_CPU_THREADS, num_threads);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting num threads: %d\n", status);
    goto DONE;
  }

  status = cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, time_limit);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting time limit: %d\n", status);
    goto DONE;
  }

  status = cuOptSetFloatParameter(settings, CUOPT_WORK_LIMIT, work_limit);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting work limit: %d\n", status);
    goto DONE;
  }

  int seed = rand();
  printf("Seed: %d\n", seed);

  for (run = 0; run < num_runs; run++) {
    cuOptSolution solution = NULL;
    cuopt_float_t objective;
    cuopt_int_t termination_status;

    status = cuOptSetIntegerParameter(settings, CUOPT_RANDOM_SEED, seed);
    if (status != CUOPT_SUCCESS) {
      printf("Error setting seed: %d\n", status);
      goto DONE;
    }

    status = cuOptSolve(problem, settings, &solution);
    if (status != CUOPT_SUCCESS) {
      printf("Error solving problem on run %d: %d\n", run, status);
      cuOptDestroySolution(&solution);
      goto DONE;
    }

    status = cuOptGetObjectiveValue(solution, &objective);
    if (status != CUOPT_SUCCESS) {
      printf("Error getting objective value on run %d: %d\n", run, status);
      cuOptDestroySolution(&solution);
      goto DONE;
    }

    status = cuOptGetTerminationStatus(solution, &termination_status);
    if (status != CUOPT_SUCCESS) {
      printf("Error getting termination status on run %d: %d\n", run, status);
      cuOptDestroySolution(&solution);
      goto DONE;
    }

    if (termination_status != CUOPT_TERMINATION_STATUS_OPTIMAL &&
        termination_status != CUOPT_TERMINATION_STATUS_TIME_LIMIT &&
        termination_status != CUOPT_TERMINATION_STATUS_FEASIBLE_FOUND) {
      printf("Run %d: status=%s (%d), unexpected termination status\n",
             run,
             termination_status_to_string(termination_status),
             termination_status);
      status = CUOPT_VALIDATION_ERROR;
      cuOptDestroySolution(&solution);
      goto DONE;
    }

    printf("Run %d: status=%s (%d), objective=%f\n",
           run,
           termination_status_to_string(termination_status),
           termination_status,
           objective);

    if (run == 0) {
      first_objective = objective;
      first_status    = termination_status;
    } else {
      if (first_status != termination_status) {
        printf("Determinism failure: run %d termination status %d differs from run 0 status %d\n",
               run,
               termination_status,
               first_status);
        status = CUOPT_VALIDATION_ERROR;
        cuOptDestroySolution(&solution);
        goto DONE;
      }
      if (first_objective != objective) {
        printf("Determinism failure: run %d objective %f differs from run 0 objective %f\n",
               run,
               objective,
               first_objective);
        status = CUOPT_VALIDATION_ERROR;
        cuOptDestroySolution(&solution);
        goto DONE;
      }
    }
    cuOptDestroySolution(&solution);
  }

  printf("Deterministic B&B test PASSED: all %d runs produced identical results\n", num_runs);

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  return status;
}

/**
 * Test that calling MIP-only methods on LP solution returns CUOPT_INVALID_ARGUMENT.
 * Uses a tiny inline LP (no file I/O):
 *   min  x + 2y   s.t.  x + y <= 10,  0 <= x,y <= 100
 */
cuopt_int_t test_lp_solution_mip_methods()
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
  cuopt_int_t status;
  cuopt_float_t mip_gap;
  cuopt_float_t solution_bound;

  cuopt_float_t obj[]   = {1.0, 2.0};
  cuopt_int_t offsets[] = {0, 2};
  cuopt_int_t indices[] = {0, 1};
  cuopt_float_t vals[]  = {1.0, 1.0};
  char sense[]          = {CUOPT_LESS_THAN};
  cuopt_float_t rhs[]   = {10.0};
  cuopt_float_t lb[]    = {0.0, 0.0};
  cuopt_float_t ub[]    = {100.0, 100.0};
  char vtypes[]         = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  printf("Testing LP solution with MIP-only methods...\n");

  status = cuOptCreateProblem(
    1, 2, CUOPT_MINIMIZE, 0.0, obj, offsets, indices, vals, sense, rhs, lb, ub, vtypes, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating LP problem: %d\n", status);
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving LP: %d\n", status);
    goto DONE;
  }

  /* Calling get_mip_gap on LP solution should return CUOPT_INVALID_ARGUMENT */
  status = cuOptGetMIPGap(solution, &mip_gap);
  if (status != CUOPT_INVALID_ARGUMENT) {
    printf("Error: cuOptGetMIPGap on LP should return CUOPT_INVALID_ARGUMENT, got %d\n", status);
    status = -1;
    goto DONE;
  }

  /* Calling get_solution_bound on LP solution should return CUOPT_INVALID_ARGUMENT */
  status = cuOptGetSolutionBound(solution, &solution_bound);
  if (status != CUOPT_INVALID_ARGUMENT) {
    printf("Error: cuOptGetSolutionBound on LP should return CUOPT_INVALID_ARGUMENT, got %d\n",
           status);
    status = -1;
    goto DONE;
  }

  printf("LP solution MIP methods test passed\n");
  status = CUOPT_SUCCESS;

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);
  return status;
}

/**
 * Test that calling LP-only methods on MIP solution returns CUOPT_INVALID_ARGUMENT.
 * Uses a tiny inline MIP (no file I/O):
 *   max  3x + 5y   s.t.  x + 2y <= 4,  x,y binary
 */
cuopt_int_t test_mip_solution_lp_methods()
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
  cuopt_int_t status;
  cuopt_float_t dual_objective;
  cuopt_float_t dual_solution[1];
  cuopt_float_t reduced_costs[2];

  cuopt_float_t obj[]   = {3.0, 5.0};
  cuopt_int_t offsets[] = {0, 2};
  cuopt_int_t indices[] = {0, 1};
  cuopt_float_t vals[]  = {1.0, 2.0};
  char sense[]          = {CUOPT_LESS_THAN};
  cuopt_float_t rhs[]   = {4.0};
  cuopt_float_t lb[]    = {0.0, 0.0};
  cuopt_float_t ub[]    = {1.0, 1.0};
  char vtypes[]         = {CUOPT_INTEGER, CUOPT_INTEGER};

  printf("Testing MIP solution with LP-only methods...\n");

  status = cuOptCreateProblem(
    1, 2, CUOPT_MAXIMIZE, 0.0, obj, offsets, indices, vals, sense, rhs, lb, ub, vtypes, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating MIP problem: %d\n", status);
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving MIP: %d\n", status);
    goto DONE;
  }

  /* Calling get_dual_objective_value on MIP solution should return CUOPT_INVALID_ARGUMENT */
  status = cuOptGetDualObjectiveValue(solution, &dual_objective);
  if (status != CUOPT_INVALID_ARGUMENT) {
    printf(
      "Error: cuOptGetDualObjectiveValue on MIP should return CUOPT_INVALID_ARGUMENT, got %d\n",
      status);
    status = -1;
    goto DONE;
  }

  /* Calling get_dual_solution on MIP solution should return CUOPT_INVALID_ARGUMENT */
  status = cuOptGetDualSolution(solution, dual_solution);
  if (status != CUOPT_INVALID_ARGUMENT) {
    printf("Error: cuOptGetDualSolution on MIP should return CUOPT_INVALID_ARGUMENT, got %d\n",
           status);
    status = -1;
    goto DONE;
  }

  /* Calling get_reduced_costs on MIP solution should return CUOPT_INVALID_ARGUMENT */
  status = cuOptGetReducedCosts(solution, reduced_costs);
  if (status != CUOPT_INVALID_ARGUMENT) {
    printf("Error: cuOptGetReducedCosts on MIP should return CUOPT_INVALID_ARGUMENT, got %d\n",
           status);
    status = -1;
    goto DONE;
  }

  printf("MIP solution LP methods test passed\n");
  status = CUOPT_SUCCESS;

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);
  return status;
}

/**
 * Test CPU-only execution with CUDA_VISIBLE_DEVICES="" and remote execution enabled.
 * This simulates a CPU host without GPU access.
 * Note: Environment variables must be set before calling this function.
 */
cuopt_int_t test_cpu_only_execution(const char* filename)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
  cuopt_int_t status;
  cuopt_int_t termination_status;
  cuopt_float_t objective_value;
  cuopt_float_t solve_time;
  cuopt_int_t num_variables;
  cuopt_int_t num_constraints;
  cuopt_float_t* primal_solution = NULL;

  printf("Testing CPU-only execution (simulated remote mode)...\n");
  printf("  CUDA_VISIBLE_DEVICES=%s\n",
         getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "(not set)");
  printf("  CUOPT_REMOTE_HOST=%s\n",
         getenv("CUOPT_REMOTE_HOST") ? getenv("CUOPT_REMOTE_HOST") : "(not set)");

  status = cuOptReadProblem(filename, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error reading problem: %d\n", status);
    goto DONE;
  }

  status = cuOptGetNumVariables(problem, &num_variables);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting num variables: %d\n", status);
    goto DONE;
  }

  status = cuOptGetNumConstraints(problem, &num_constraints);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting num constraints: %d\n", status);
    goto DONE;
  }

  printf("  Problem: %d variables, %d constraints\n", num_variables, num_constraints);

  if (num_variables > 0) {
    primal_solution = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
    if (primal_solution == NULL) {
      printf("Error allocating primal solution\n");
      status = -1;
      goto DONE;
    }
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_PDLP);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting method: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem: %d\n", status);
    goto DONE;
  }

  /* Verify we can retrieve all solution properties without CUDA errors */
  status = cuOptGetTerminationStatus(solution, &termination_status);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status: %d\n", status);
    goto DONE;
  }

  status = cuOptGetObjectiveValue(solution, &objective_value);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

  status = cuOptGetSolveTime(solution, &solve_time);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solve time: %d\n", status);
    goto DONE;
  }

  status = cuOptGetPrimalSolution(solution, primal_solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting primal solution: %d\n", status);
    goto DONE;
  }

  printf("CPU-only execution test passed\n");
  printf("  Termination status: %s\n", termination_status_to_string(termination_status));
  printf("  Objective value: %f\n", objective_value);
  printf("  Solve time: %f\n", solve_time);
  if (num_variables > 0) { printf("  Primal solution[0]: %f\n", primal_solution[0]); }

  status = CUOPT_SUCCESS;

DONE:
  free(primal_solution);
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);
  return status;
}

/**
 * Test CPU-only MIP execution with CUDA_VISIBLE_DEVICES="" and remote execution enabled.
 */
cuopt_int_t test_cpu_only_mip_execution(const char* filename)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
  cuopt_int_t status;
  cuopt_int_t termination_status;
  cuopt_float_t objective_value;
  cuopt_float_t solve_time;
  cuopt_float_t mip_gap;
  cuopt_int_t num_variables;
  cuopt_float_t* primal_solution = NULL;

  printf("Testing CPU-only MIP execution (simulated remote mode)...\n");

  status = cuOptReadProblem(filename, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error reading MIP problem: %d\n", status);
    goto DONE;
  }

  status = cuOptGetNumVariables(problem, &num_variables);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting num variables: %d\n", status);
    goto DONE;
  }

  primal_solution = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  if (!primal_solution && num_variables > 0) {
    printf("Error: malloc failed for primal_solution\n");
    status = -1;
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, 60.0);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting time limit: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving MIP: %d\n", status);
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, &termination_status);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status: %d\n", status);
    goto DONE;
  }

  status = cuOptGetObjectiveValue(solution, &objective_value);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

  status = cuOptGetSolveTime(solution, &solve_time);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solve time: %d\n", status);
    goto DONE;
  }

  status = cuOptGetMIPGap(solution, &mip_gap);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting MIP gap: %d\n", status);
    goto DONE;
  }

  status = cuOptGetPrimalSolution(solution, primal_solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting primal solution: %d\n", status);
    goto DONE;
  }

  printf("CPU-only MIP execution test passed\n");
  printf("  Termination status: %s\n", termination_status_to_string(termination_status));
  printf("  Objective value: %f\n", objective_value);
  printf("  MIP gap: %f\n", mip_gap);
  printf("  Solve time: %f\n", solve_time);

  status = CUOPT_SUCCESS;

DONE:
  free(primal_solution);
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);
  return status;
}

cuopt_int_t test_pdlp_precision_mixed(const char* filename,
                                      cuopt_int_t* termination_status_ptr,
                                      cuopt_float_t* objective_ptr)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
  cuopt_int_t status;
  cuopt_int_t termination_status = -1;
  cuopt_float_t objective_value;

  status = cuOptReadProblem(filename, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error reading problem\n");
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings\n");
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_PDLP);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting method\n");
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_PDLP_PRECISION, CUOPT_PDLP_MIXED_PRECISION);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting pdlp_precision\n");
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem with pdlp_precision=mixed\n");
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, &termination_status);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status\n");
    goto DONE;
  }
  *termination_status_ptr = termination_status;

  status = cuOptGetObjectiveValue(solution, &objective_value);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value\n");
    goto DONE;
  }
  *objective_ptr = objective_value;

  printf("PDLP precision=mixed test passed: status=%s, objective=%f\n",
         termination_status_to_string(termination_status),
         objective_value);

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);
  return status;
}

cuopt_int_t test_pdlp_precision_single(const char* filename,
                                       cuopt_int_t* termination_status_ptr,
                                       cuopt_float_t* objective_ptr)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
  cuopt_int_t status;
  cuopt_int_t termination_status = -1;
  cuopt_float_t objective_value;

  status = cuOptReadProblem(filename, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error reading problem\n");
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings\n");
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_PDLP);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting method\n");
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_PDLP_PRECISION, CUOPT_PDLP_SINGLE_PRECISION);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting pdlp_precision\n");
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem with pdlp_precision=single\n");
    goto DONE;
  }

  status = cuOptGetTerminationStatus(solution, &termination_status);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting termination status\n");
    goto DONE;
  }
  *termination_status_ptr = termination_status;

  status = cuOptGetObjectiveValue(solution, &objective_value);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value\n");
    goto DONE;
  }
  *objective_ptr = objective_value;

  printf("PDLP precision=single test passed: status=%s, objective=%f\n",
         termination_status_to_string(termination_status),
         objective_value);

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);
  return status;
}

/**
 * Exercise read-only C API on a CPU-backed problem (CUDA_VISIBLE_DEVICES="" and no remote).
 * Caller must set CUDA_VISIBLE_DEVICES="" before invoking.
 */
cuopt_int_t test_cpu_host_read_problem_api(const char* filename)
{
  cuOptOptimizationProblem problem = NULL;
  cuopt_int_t status;
  cuopt_int_t num_variables    = 0;
  cuopt_int_t num_constraints  = 0;
  cuopt_int_t nnz              = 0;
  cuopt_int_t objective_sense  = 0;
  cuopt_float_t objective_offset = 0;
  cuopt_float_t* objective_coefficients = NULL;
  cuopt_float_t* var_lower_bounds       = NULL;
  cuopt_float_t* var_upper_bounds       = NULL;
  cuopt_int_t* row_offsets              = NULL;
  cuopt_int_t* column_indices           = NULL;
  cuopt_float_t* values                 = NULL;
  char* constraint_sense                = NULL;
  cuopt_float_t* rhs                    = NULL;
  cuopt_int_t is_mip                    = 0;

  status = cuOptReadProblem(filename, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error reading problem on CPU host: %d\n", status);
    goto DONE;
  }

  status = cuOptGetNumVariables(problem, &num_variables);
  if (status != CUOPT_SUCCESS || num_variables != 32) {
    printf("Unexpected num variables on CPU host (status=%d, n=%d)\n", status, num_variables);
    status = CUOPT_VALIDATION_ERROR;
    goto DONE;
  }

  status = cuOptGetNumConstraints(problem, &num_constraints);
  if (status != CUOPT_SUCCESS || num_constraints != 27) {
    printf("Unexpected num constraints on CPU host (status=%d, n=%d)\n", status, num_constraints);
    status = CUOPT_VALIDATION_ERROR;
    goto DONE;
  }

  status = cuOptGetNumNonZeros(problem, &nnz);
  if (status != CUOPT_SUCCESS || nnz <= 0) {
    printf("Unexpected nnz on CPU host (status=%d, nnz=%d)\n", status, nnz);
    status = CUOPT_VALIDATION_ERROR;
    goto DONE;
  }

  status = cuOptGetObjectiveSense(problem, &objective_sense);
  if (status != CUOPT_SUCCESS || objective_sense != CUOPT_MINIMIZE) {
    printf("Unexpected objective sense on CPU host (status=%d, sense=%d)\n",
           status,
           objective_sense);
    status = CUOPT_VALIDATION_ERROR;
    goto DONE;
  }

  status = cuOptGetObjectiveOffset(problem, &objective_offset);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective offset on CPU host: %d\n", status);
    goto DONE;
  }

  objective_coefficients = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  var_lower_bounds       = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  var_upper_bounds       = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  row_offsets            = (cuopt_int_t*)malloc((num_constraints + 1) * sizeof(cuopt_int_t));
  column_indices         = (cuopt_int_t*)malloc(nnz * sizeof(cuopt_int_t));
  values                 = (cuopt_float_t*)malloc(nnz * sizeof(cuopt_float_t));
  constraint_sense       = (char*)malloc(num_constraints * sizeof(char));
  rhs                    = (cuopt_float_t*)malloc(num_constraints * sizeof(cuopt_float_t));
  if (objective_coefficients == NULL || var_lower_bounds == NULL || var_upper_bounds == NULL ||
      row_offsets == NULL || column_indices == NULL || values == NULL || constraint_sense == NULL ||
      rhs == NULL) {
    printf("Out of memory allocating CPU host API buffers\n");
    status = CUOPT_OUT_OF_MEMORY;
    goto DONE;
  }

  status = cuOptGetObjectiveCoefficients(problem, objective_coefficients);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective coefficients on CPU host: %d\n", status);
    goto DONE;
  }

  status = cuOptGetVariableLowerBounds(problem, var_lower_bounds);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting variable lower bounds on CPU host: %d\n", status);
    goto DONE;
  }

  status = cuOptGetVariableUpperBounds(problem, var_upper_bounds);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting variable upper bounds on CPU host: %d\n", status);
    goto DONE;
  }

  status = cuOptGetConstraintMatrix(problem, row_offsets, column_indices, values);
  if (status != CUOPT_SUCCESS || row_offsets[0] != 0 || row_offsets[num_constraints] != nnz) {
    printf("Unexpected constraint matrix on CPU host (status=%d)\n", status);
    status = CUOPT_VALIDATION_ERROR;
    goto DONE;
  }

  status = cuOptGetConstraintSense(problem, constraint_sense);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting constraint sense on CPU host: %d\n", status);
    goto DONE;
  }

  status = cuOptGetConstraintRightHandSide(problem, rhs);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting constraint rhs on CPU host: %d\n", status);
    goto DONE;
  }

  status = cuOptIsMIP(problem, &is_mip);
  if (status != CUOPT_SUCCESS || is_mip != 0) {
    printf("Expected LP problem on CPU host (status=%d, is_mip=%d)\n", status, is_mip);
    status = CUOPT_VALIDATION_ERROR;
    goto DONE;
  }

  printf("CPU host read-only C API test passed (%d vars, %d cons, nnz=%d)\n",
         num_variables,
         num_constraints,
         nnz);

DONE:
  free(objective_coefficients);
  free(var_lower_bounds);
  free(var_upper_bounds);
  free(row_offsets);
  free(column_indices);
  free(values);
  free(constraint_sense);
  free(rhs);
  cuOptDestroyProblem(&problem);
  return status;
}

/**
 * Build a small MIP via cuOptCreateProblem on a CPU-backed host and verify getters (no solve).
 * Caller must set CUDA_VISIBLE_DEVICES="" before invoking.
 */
cuopt_int_t test_cpu_host_create_problem_api()
{
  cuOptOptimizationProblem problem = NULL;
  cuopt_int_t status;

  enum { k_num_items = 8, k_num_constraints = 1 };
  const cuopt_int_t num_variables   = k_num_items;
  const cuopt_int_t num_constraints = k_num_constraints;
  const cuopt_int_t nnz             = k_num_items;
  const cuopt_float_t max_weight    = 102;
  cuopt_float_t value[]             = {15, 100, 90, 60, 40, 15, 10, 1};
  cuopt_float_t weight[]            = {2, 20, 20, 30, 40, 30, 60, 10};
  cuopt_int_t row_offsets[]         = {0, k_num_items};
  cuopt_int_t column_indices[k_num_items];
  cuopt_float_t rhs[]               = {max_weight};
  char constraint_sense[]           = {CUOPT_LESS_THAN};
  cuopt_float_t lower_bounds[k_num_items];
  cuopt_float_t upper_bounds[k_num_items];
  char variable_types[k_num_items];
  cuopt_int_t objective_sense         = CUOPT_MAXIMIZE;
  cuopt_float_t objective_offset      = 0;
  cuopt_int_t is_mip                  = 0;

  for (cuopt_int_t j = 0; j < k_num_items; j++) {
    column_indices[j] = j;
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
    printf("Error creating optimization problem on CPU host: %d\n", status);
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
    printf("Error checking CPU host created problem\n");
    goto DONE;
  }

  status = cuOptIsMIP(problem, &is_mip);
  if (status != CUOPT_SUCCESS || is_mip != 1) {
    printf("Expected MIP on CPU host (status=%d, is_mip=%d)\n", status, is_mip);
    status = CUOPT_VALIDATION_ERROR;
    goto DONE;
  }

  printf("CPU host create-problem C API test passed\n");

DONE:
  cuOptDestroyProblem(&problem);
  return status;
}

/**
 * Read a problem as a GPU-backed handle (remote env must be unset), then enable
 * remote execution and verify cuOptSolve rejects the GPU problem.
 */
cuopt_int_t test_gpu_problem_remote_after_create(const char* filename)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
  cuopt_int_t status;

  const char* orig_remote_host = getenv("CUOPT_REMOTE_HOST");
  const char* orig_remote_port = getenv("CUOPT_REMOTE_PORT");
  const int host_was_set       = (orig_remote_host != NULL);
  const int port_was_set       = (orig_remote_port != NULL);
  char saved_remote_host[256]  = "";
  char saved_remote_port[64]   = "";

  if (host_was_set) {
    strncpy(saved_remote_host, orig_remote_host, sizeof(saved_remote_host) - 1);
    saved_remote_host[sizeof(saved_remote_host) - 1] = '\0';
  }
  if (port_was_set) {
    strncpy(saved_remote_port, orig_remote_port, sizeof(saved_remote_port) - 1);
    saved_remote_port[sizeof(saved_remote_port) - 1] = '\0';
  }

  unsetenv("CUOPT_REMOTE_HOST");
  unsetenv("CUOPT_REMOTE_PORT");

  printf("Testing GPU problem rejects remote solve after create...\n");
  printf("  (read with remote unset, then set CUOPT_REMOTE_* before solve)\n");

  status = cuOptReadProblem(filename, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error reading problem: %d\n", status);
    goto DONE;
  }

  setenv("CUOPT_REMOTE_HOST", "localhost", 1);
  setenv("CUOPT_REMOTE_PORT", "18500", 1);

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSetParameter(settings, "log_to_console", "true");
  if (status != CUOPT_SUCCESS) {
    printf("Error setting log_to_console: %d\n", status);
    goto DONE;
  }

  status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_PDLP);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting method: %d\n", status);
    goto DONE;
  }

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_VALIDATION_ERROR) {
    printf("Expected CUOPT_VALIDATION_ERROR (%d), got %d\n", CUOPT_VALIDATION_ERROR, status);
    status = -1;
    goto DONE;
  }

  printf("GPU problem correctly rejected remote solve with CUOPT_VALIDATION_ERROR\n");
  status = CUOPT_SUCCESS;

DONE:
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  if (host_was_set) {
    setenv("CUOPT_REMOTE_HOST", saved_remote_host, 1);
  } else {
    unsetenv("CUOPT_REMOTE_HOST");
  }
  if (port_was_set) {
    setenv("CUOPT_REMOTE_PORT", saved_remote_port, 1);
  } else {
    unsetenv("CUOPT_REMOTE_PORT");
  }

  return status;
}

static int doubles_equal(const cuopt_float_t* a, const cuopt_float_t* b, cuopt_int_t n)
{
  cuopt_int_t i;
  for (i = 0; i < n; ++i) {
    if (a[i] != b[i]) { return 0; }
  }
  return 1;
}

static int ints_equal(const cuopt_int_t* a, const cuopt_int_t* b, cuopt_int_t n)
{
  cuopt_int_t i;
  for (i = 0; i < n; ++i) {
    if (a[i] != b[i]) { return 0; }
  }
  return 1;
}

/* Report a failing C API call. Returns the status so callers can do:
 *   status = check_ok(cuOptSomeCall(...), "what"); if (status != CUOPT_SUCCESS) goto DONE; */
static cuopt_int_t check_ok(cuopt_int_t status, const char* what)
{
  if (status != CUOPT_SUCCESS) { printf("FAILED (%d): %s\n", (int)status, what); }
  return status;
}

/* Report a failing ground-truth comparison. Returns CUOPT_SUCCESS when cond holds, otherwise
 * CUOPT_VALIDATION_ERROR, so callers can do:
 *   status = check_true(got == expected, "what"); if (status != CUOPT_SUCCESS) goto DONE; */
static cuopt_int_t check_true(int cond, const char* what)
{
  if (!cond) { printf("ground-truth failure: %s\n", what); }
  return cond ? CUOPT_SUCCESS : CUOPT_VALIDATION_ERROR;
}

/* One sparse matrix entry, used to compare CSR and CSC as an unordered (row,col,value) multiset. */
typedef struct {
  cuopt_int_t row;
  cuopt_int_t col;
  cuopt_float_t val;
} coo_triple_t;

/* Total order by (row, col, value) so two equivalent matrices sort to identical sequences. */
static int compare_coo(const void* a, const void* b)
{
  const coo_triple_t* ta = (const coo_triple_t*)a;
  const coo_triple_t* tb = (const coo_triple_t*)b;
  if (ta->row != tb->row) { return ta->row < tb->row ? -1 : 1; }
  if (ta->col != tb->col) { return ta->col < tb->col ? -1 : 1; }
  if (ta->val != tb->val) { return ta->val < tb->val ? -1 : 1; }
  return 0;
}

/*
 * Build a small mixed-integer LP by hand with cuOptCreateProblem, then read every getter back and
 * compare against the exact values we constructed. Ground truth comes from the create call itself,
 * so there is no dependence on a parser or a dataset file, and it works on either backend. This
 * covers every attribute cuOptCreateProblem can set; the ranged-only, quadratic-only, and name-only
 * attributes are covered by test_problem_attributes_ranged, _quadratic, and _names respectively.
 */
cuopt_int_t test_problem_attributes_created(void)
{
  cuOptOptimizationProblem problem = NULL;
  cuopt_int_t status               = CUOPT_SUCCESS;

  /* Known construction: 2 constraints x 3 variables, mixed integer.
   *   A = [ 1 0 2 ]   sense=L rhs=10
   *       [ 0 3 4 ]   sense=G rhs=20
   *   obj = 5 + [1,2,3].x   var types = [I,C,I] */
  const cuopt_int_t num_constraints            = 2;
  const cuopt_int_t num_variables              = 3;
  const cuopt_int_t nnz                        = 4;
  const cuopt_float_t objective_offset         = 5.0;
  const cuopt_float_t objective_coefficients[3] = {1.0, 2.0, 3.0};
  const cuopt_int_t row_offsets[3]             = {0, 2, 4};
  const cuopt_int_t col_indices[4]             = {0, 2, 1, 2};
  const cuopt_float_t matrix_values[4]         = {1.0, 2.0, 3.0, 4.0};
  const char constraint_sense[2]               = {CUOPT_LESS_THAN, CUOPT_GREATER_THAN};
  const cuopt_float_t rhs[2]                    = {10.0, 20.0};
  const cuopt_float_t lower_bounds[3]          = {0.0, 0.0, 0.0};
  const cuopt_float_t upper_bounds[3]          = {100.0, 100.0, 100.0};
  const char variable_types[3]                 = {CUOPT_INTEGER, CUOPT_CONTINUOUS, CUOPT_INTEGER};

  cuopt_float_t fbuf[3];
  char cbuf[3];
  cuopt_int_t ival    = 0;
  cuopt_float_t fval  = 0.0;
  cuopt_int_t csr_off[3];
  cuopt_int_t csr_col[4];
  cuopt_float_t csr_val[4];
  cuopt_int_t csc_off[4];
  cuopt_int_t csc_row[4];
  cuopt_float_t csc_val[4];
  coo_triple_t exp_tr[4];
  coo_triple_t got_tr[4];
  cuopt_int_t i = 0, k = 0;
  const char* names_probe[3];

  status = check_ok(cuOptCreateProblem(num_constraints,
                                       num_variables,
                                       CUOPT_MINIMIZE,
                                       objective_offset,
                                       objective_coefficients,
                                       row_offsets,
                                       col_indices,
                                       matrix_values,
                                       constraint_sense,
                                       rhs,
                                       lower_bounds,
                                       upper_bounds,
                                       variable_types,
                                       &problem),
                    "cuOptCreateProblem");
  if (status != CUOPT_SUCCESS) goto DONE;

  /* --- dimensions and objective sense: generic getters vs the construction --- */
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_NUM_VARIABLES, &ival),
                    "get num_variables");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == num_variables, "num_variables");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_NUM_CONSTRAINTS, &ival),
                    "get num_constraints");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == num_constraints, "num_constraints");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_NUM_NONZEROS, &ival),
                    "get num_nonzeros");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == nnz, "num_nonzeros");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_OBJECTIVE_SENSE, &ival),
                    "get objective_sense");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == CUOPT_MINIMIZE, "objective_sense");
  if (status != CUOPT_SUCCESS) goto DONE;

  /* --- scalar attributes with no dedicated getter: verify against the known construction --- */
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_NUM_INTEGERS, &ival),
                    "get num_integers");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == 2, "num_integers");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_PROBLEM_CATEGORY, &ival),
                    "get problem_category");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == 1 /* MIP */, "problem_category");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_IS_MIP, &ival), "get is_mip");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == 1, "is_mip");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_HAS_QUADRATIC_OBJECTIVE, &ival),
                    "get has_quadratic_objective");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == 0, "has_quadratic_objective");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_HAS_QUADRATIC_CONSTRAINTS, &ival),
                    "get has_quadratic_constraints");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == 0, "has_quadratic_constraints");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemFloatAttribute(problem, CUOPT_ATTR_OBJECTIVE_SCALING_FACTOR, &fval),
                    "get objective_scaling_factor");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(fval == 1.0, "objective_scaling_factor"); /* create leaves the default of 1 */
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemFloatAttribute(problem, CUOPT_ATTR_OBJECTIVE_OFFSET, &fval),
                    "get objective_offset");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(fval == objective_offset, "objective_offset");
  if (status != CUOPT_SUCCESS) goto DONE;

  /* --- array getters: compare against the exact values we constructed with --- */
  status = check_ok(cuOptGetProblemFloatArrayAttribute(
                      problem, CUOPT_ARRAY_ATTR_OBJECTIVE_COEFFICIENTS, fbuf, num_variables),
                    "get objective_coefficients");
  if (status != CUOPT_SUCCESS) goto DONE;
  status =
    check_true(doubles_equal(fbuf, objective_coefficients, num_variables), "objective_coefficients");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemFloatArrayAttribute(
                      problem, CUOPT_ARRAY_ATTR_VARIABLE_LOWER_BOUNDS, fbuf, num_variables),
                    "get variable_lower_bounds");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(doubles_equal(fbuf, lower_bounds, num_variables), "variable_lower_bounds");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemFloatArrayAttribute(
                      problem, CUOPT_ARRAY_ATTR_VARIABLE_UPPER_BOUNDS, fbuf, num_variables),
                    "get variable_upper_bounds");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(doubles_equal(fbuf, upper_bounds, num_variables), "variable_upper_bounds");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(
    cuOptGetProblemFloatArrayAttribute(problem, CUOPT_ARRAY_ATTR_CONSTRAINT_RHS, fbuf, num_constraints),
    "get constraint_rhs");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(doubles_equal(fbuf, rhs, num_constraints), "constraint_rhs");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(
    cuOptGetProblemCharArrayAttribute(problem, CUOPT_ARRAY_ATTR_CONSTRAINT_SENSE, cbuf, num_constraints),
    "get constraint_sense");
  if (status != CUOPT_SUCCESS) goto DONE;
  status =
    check_true(memcmp(cbuf, constraint_sense, (size_t)num_constraints) == 0, "constraint_sense");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(
    cuOptGetProblemCharArrayAttribute(problem, CUOPT_ARRAY_ATTR_VARIABLE_TYPES, cbuf, num_variables),
    "get variable_types");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(memcmp(cbuf, variable_types, (size_t)num_variables) == 0, "variable_types");
  if (status != CUOPT_SUCCESS) goto DONE;

  /* --- CSR values: the getter must return exactly the CSR we created the problem with --- */
  status = check_ok(cuOptGetConstraintMatrixCSR(problem, csr_off, csr_col, csr_val), "get CSR matrix");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ints_equal(csr_off, row_offsets, num_constraints + 1) &&
                        ints_equal(csr_col, col_indices, nnz) &&
                        doubles_equal(csr_val, matrix_values, nnz),
                      "constraint_matrix_csr");
  if (status != CUOPT_SUCCESS) goto DONE;

  /* --- CSC values: must equal the transpose of the known CSR input (row,col,value multiset) --- */
  status = check_ok(cuOptGetConstraintMatrixCSC(problem, csc_off, csc_row, csc_val), "get CSC matrix");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(csc_off[0] == 0 && csc_off[num_variables] == nnz, "csc_offsets");
  if (status != CUOPT_SUCCESS) goto DONE;
  for (i = 0; i < num_constraints; ++i) {
    for (k = row_offsets[i]; k < row_offsets[i + 1]; ++k) {
      exp_tr[k].row = i;
      exp_tr[k].col = col_indices[k];
      exp_tr[k].val = matrix_values[k];
    }
  }
  for (i = 0; i < num_variables; ++i) {
    for (k = csc_off[i]; k < csc_off[i + 1]; ++k) {
      got_tr[k].row = csc_row[k];
      got_tr[k].col = i;
      got_tr[k].val = csc_val[k];
    }
  }
  qsort(exp_tr, (size_t)nnz, sizeof(coo_triple_t), compare_coo);
  qsort(got_tr, (size_t)nnz, sizeof(coo_triple_t), compare_coo);
  {
    int csc_ok = 1;
    for (k = 0; k < nnz; ++k) {
      if (exp_tr[k].row != got_tr[k].row || exp_tr[k].col != got_tr[k].col ||
          exp_tr[k].val != got_tr[k].val) {
        csc_ok = 0;
        break;
      }
    }
    status = check_true(csc_ok, "constraint_matrix_csc (transpose of CSR)");
    if (status != CUOPT_SUCCESS) goto DONE;
  }

  /* cuOptCreateProblem does not set variable/row names, so the string-array getter has nothing to
     return; asking for num_variables names must be rejected. Name *values* are verified from a
     parsed file in test_problem_attributes_names. */
  status = check_true(cuOptGetProblemStringArrayAttribute(
                        problem, CUOPT_STRING_ARRAY_VARIABLE_NAMES, names_probe, num_variables) ==
                        CUOPT_INVALID_ARGUMENT,
                      "unnamed problem rejects name getter");
  if (status != CUOPT_SUCCESS) goto DONE;

DONE:
  cuOptDestroyProblem(&problem);
  return status;
}

/*
 * Ranged rows can only be built with cuOptCreateRangedProblem, so this is the create-based ground
 * truth for the constraint lower/upper bound getters (the sense+rhs create path leaves those empty).
 * Build a 2x2 ranged LP and read the row bounds back.
 *
 *    1 <= x0 + x1 <= 10
 *    0 <=      x0 <=  5
 */
cuopt_int_t test_problem_attributes_ranged(void)
{
  cuOptOptimizationProblem problem = NULL;
  cuopt_int_t status               = CUOPT_SUCCESS;

  const cuopt_int_t num_constraints              = 2;
  const cuopt_int_t num_variables               = 2;
  const cuopt_int_t nnz                         = 3;
  const cuopt_float_t objective_coefficients[2] = {1.0, 1.0};
  const cuopt_int_t row_offsets[3]              = {0, 2, 3};
  const cuopt_int_t col_indices[3]              = {0, 1, 0};
  const cuopt_float_t matrix_values[3]          = {1.0, 1.0, 1.0};
  const cuopt_float_t constraint_lower_bounds[2] = {1.0, 0.0};
  const cuopt_float_t constraint_upper_bounds[2] = {10.0, 5.0};
  const cuopt_float_t variable_lower_bounds[2]  = {0.0, 0.0};
  const cuopt_float_t variable_upper_bounds[2]  = {100.0, 100.0};
  const char variable_types[2]                  = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  cuopt_int_t ival = 0;
  cuopt_float_t fbuf[2];

  status = check_ok(cuOptCreateRangedProblem(num_constraints,
                                             num_variables,
                                             CUOPT_MINIMIZE,
                                             0.0,
                                             objective_coefficients,
                                             row_offsets,
                                             col_indices,
                                             matrix_values,
                                             constraint_lower_bounds,
                                             constraint_upper_bounds,
                                             variable_lower_bounds,
                                             variable_upper_bounds,
                                             variable_types,
                                             &problem),
                    "cuOptCreateRangedProblem");
  if (status != CUOPT_SUCCESS) goto DONE;

  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_NUM_CONSTRAINTS, &ival),
                    "get num_constraints");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == num_constraints, "num_constraints");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_NUM_VARIABLES, &ival),
                    "get num_variables");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == num_variables, "num_variables");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_NUM_NONZEROS, &ival),
                    "get num_nonzeros");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == nnz, "num_nonzeros");
  if (status != CUOPT_SUCCESS) goto DONE;

  status = check_ok(cuOptGetProblemFloatArrayAttribute(
                      problem, CUOPT_ARRAY_ATTR_CONSTRAINT_LOWER_BOUNDS, fbuf, num_constraints),
                    "get constraint_lower_bounds");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(doubles_equal(fbuf, constraint_lower_bounds, num_constraints),
                      "constraint_lower_bounds");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemFloatArrayAttribute(
                      problem, CUOPT_ARRAY_ATTR_CONSTRAINT_UPPER_BOUNDS, fbuf, num_constraints),
                    "get constraint_upper_bounds");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(doubles_equal(fbuf, constraint_upper_bounds, num_constraints),
                      "constraint_upper_bounds");
  if (status != CUOPT_SUCCESS) goto DONE;

DONE:
  cuOptDestroyProblem(&problem);
  return status;
}

/*
 * Quadratic terms are added with cuOptSetQuadraticObjective / cuOptAddQuadraticConstraint, so this is
 * the create-based ground truth for the has_quadratic_objective / has_quadratic_constraints flags.
 * Start from a linear problem (both flags 0), add a quadratic objective, then a quadratic
 * constraint, and confirm each flag flips to 1.
 */
cuopt_int_t test_problem_attributes_quadratic(void)
{
  cuOptOptimizationProblem problem = NULL;
  cuopt_int_t status               = CUOPT_SUCCESS;

  const cuopt_int_t num_constraints             = 1;
  const cuopt_int_t num_variables               = 2;
  const cuopt_float_t objective_coefficients[2] = {1.0, 1.0};
  const cuopt_int_t row_offsets[2]              = {0, 2};
  const cuopt_int_t col_indices[2]              = {0, 1};
  const cuopt_float_t matrix_values[2]          = {1.0, 1.0};
  const char constraint_sense[1]                = {CUOPT_LESS_THAN};
  const cuopt_float_t rhs[1]                    = {10.0};
  const cuopt_float_t lower_bounds[2]           = {0.0, 0.0};
  const cuopt_float_t upper_bounds[2]           = {100.0, 100.0};
  const char variable_types[2]                  = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  /* Quadratic objective term  2 * x0^2  (coordinate/triplet form). */
  const cuopt_int_t q_row[1]   = {0};
  const cuopt_int_t q_col[1]   = {0};
  const cuopt_float_t q_val[1] = {2.0};

  /* Quadratic constraint  x1^2 + x0 <= 5. */
  const cuopt_int_t qc_row[1]         = {1};
  const cuopt_int_t qc_col[1]         = {1};
  const cuopt_float_t qc_val[1]       = {1.0};
  const cuopt_int_t qc_lin_idx[1]     = {0};
  const cuopt_float_t qc_lin_coeff[1] = {1.0};

  cuopt_int_t ival = 0;

  status = check_ok(cuOptCreateProblem(num_constraints,
                                       num_variables,
                                       CUOPT_MINIMIZE,
                                       0.0,
                                       objective_coefficients,
                                       row_offsets,
                                       col_indices,
                                       matrix_values,
                                       constraint_sense,
                                       rhs,
                                       lower_bounds,
                                       upper_bounds,
                                       variable_types,
                                       &problem),
                    "cuOptCreateProblem");
  if (status != CUOPT_SUCCESS) goto DONE;

  /* Purely linear to start: both flags must read 0. */
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_HAS_QUADRATIC_OBJECTIVE, &ival),
                    "get has_quadratic_objective");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == 0, "has_quadratic_objective before set");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_HAS_QUADRATIC_CONSTRAINTS, &ival),
                    "get has_quadratic_constraints");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == 0, "has_quadratic_constraints before add");
  if (status != CUOPT_SUCCESS) goto DONE;

  status = check_ok(cuOptSetQuadraticObjective(problem, 1, q_row, q_col, q_val),
                    "cuOptSetQuadraticObjective");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_HAS_QUADRATIC_OBJECTIVE, &ival),
                    "get has_quadratic_objective");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == 1, "has_quadratic_objective after set");
  if (status != CUOPT_SUCCESS) goto DONE;

  status = check_ok(cuOptAddQuadraticConstraint(
                      problem, 1, qc_row, qc_col, qc_val, 1, qc_lin_idx, qc_lin_coeff, CUOPT_LESS_THAN, 5.0),
                    "cuOptAddQuadraticConstraint");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_HAS_QUADRATIC_CONSTRAINTS, &ival),
                    "get has_quadratic_constraints");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == 1, "has_quadratic_constraints after add");
  if (status != CUOPT_SUCCESS) goto DONE;

DONE:
  cuOptDestroyProblem(&problem);
  return status;
}

/*
 * Variable and row names are the only attributes with no cuOptCreate / cuOptSet setter, so they are
 * the sole getters that require cuOptReadProblem. Write a tiny, hand-inspectable MPS with known
 * names, read it back, and check the name getters against ground truth. Every other attribute is
 * covered by the create-based tests above.
 */
cuopt_int_t test_problem_attributes_names(const char* mps_path)
{
  cuOptOptimizationProblem problem = NULL;
  cuopt_int_t status               = CUOPT_SUCCESS;
  FILE* f                          = NULL;

  const cuopt_int_t exp_nv     = 2;
  const cuopt_int_t exp_nc     = 2;
  const char* exp_var_names[2] = {"X1", "X2"};
  const char* exp_row_names[2] = {"C1", "C2"};

  const char* nbuf[2];
  cuopt_int_t ival = 0;
  cuopt_int_t i    = 0;

  f = fopen(mps_path, "w");
  if (f == NULL) {
    printf("names: could not open %s for writing\n", mps_path);
    return CUOPT_VALIDATION_ERROR;
  }
  fprintf(f,
          "NAME          TOY\n"
          "ROWS\n"
          " N  COST\n"
          " L  C1\n"
          " G  C2\n"
          "COLUMNS\n"
          "    X1        COST      1.0        C1        1.0\n"
          "    X1        C2        1.0\n"
          "    X2        COST      2.0        C1        3.0\n"
          "    X2        C2        1.0\n"
          "RHS\n"
          "    RHS       C1        10.0       C2        2.0\n"
          "ENDATA\n");
  fclose(f);

  status = check_ok(cuOptReadProblem(mps_path, &problem), "cuOptReadProblem");
  if (status != CUOPT_SUCCESS) goto DONE;

  /* Dimensions only to size the name buffers; their values are ground-truthed by other tests. */
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_NUM_VARIABLES, &ival),
                    "get num_variables");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == exp_nv, "num_variables");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_ok(cuOptGetProblemIntAttribute(problem, CUOPT_ATTR_NUM_CONSTRAINTS, &ival),
                    "get num_constraints");
  if (status != CUOPT_SUCCESS) goto DONE;
  status = check_true(ival == exp_nc, "num_constraints");
  if (status != CUOPT_SUCCESS) goto DONE;

  status = check_ok(
    cuOptGetProblemStringArrayAttribute(problem, CUOPT_STRING_ARRAY_VARIABLE_NAMES, nbuf, exp_nv),
    "get variable_names");
  if (status != CUOPT_SUCCESS) goto DONE;
  for (i = 0; i < exp_nv; ++i) {
    status = check_true(nbuf[i] != NULL && strcmp(nbuf[i], exp_var_names[i]) == 0, "variable_names");
    if (status != CUOPT_SUCCESS) goto DONE;
  }
  status = check_ok(
    cuOptGetProblemStringArrayAttribute(problem, CUOPT_STRING_ARRAY_ROW_NAMES, nbuf, exp_nc),
    "get row_names");
  if (status != CUOPT_SUCCESS) goto DONE;
  for (i = 0; i < exp_nc; ++i) {
    status = check_true(nbuf[i] != NULL && strcmp(nbuf[i], exp_row_names[i]) == 0, "row_names");
    if (status != CUOPT_SUCCESS) goto DONE;
  }

DONE:
  cuOptDestroyProblem(&problem);
  remove(mps_path);
  return status;
}
