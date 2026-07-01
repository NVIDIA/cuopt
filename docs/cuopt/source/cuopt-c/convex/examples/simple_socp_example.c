/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
/*
 * Simple SOCP C API Example
 *
 * Demonstrates building a second-order cone program (SOCP) with the cuOpt C API:
 * a linear problem is created with cuOptCreateProblem, then a quadratic
 * constraint is added with cuOptAddQuadraticConstraint. cuOpt detects the
 * second-order cone structure of the quadratic constraint and solves with the
 * barrier method.
 *
 * Problem:
 *   minimize    x3
 *   subject to  x1 + x2 >= 2
 *               x1^2 + x2^2 - x3^2 <= 0      (i.e. ||(x1, x2)||_2 <= x3)
 *               x3 >= 0,  x1, x2 free
 *
 * Optimal: x1 = x2 = 1, x3 = sqrt(2) ~= 1.414214, objective ~= 1.414214.
 *
 * Build:
 *   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o simple_socp_example simple_socp_example.c -lcuopt
 *
 * Run:
 *   ./simple_socp_example
 */

#include <cuopt/mathematical_optimization/cuopt_c.h>
#include <stdio.h>
#include <stdlib.h>

// Convert termination status to string
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
    default:
      return "Unknown";
  }
}

cuopt_int_t test_simple_socp()
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  cuopt_int_t num_variables   = 3;  // x1, x2, x3
  cuopt_int_t num_constraints = 1;  // linear: x1 + x2 >= 2 (the cone is added separately)

  // Linear objective: minimize x3
  cuopt_float_t objective_coefficients[] = {0.0, 0.0, 1.0};

  // Linear constraint matrix in CSR: row 0 is x1 + x2
  cuopt_int_t row_offsets[]    = {0, 2};
  cuopt_int_t column_indices[] = {0, 1};
  cuopt_float_t values[]       = {1.0, 1.0};
  char constraint_sense[]      = {CUOPT_GREATER_THAN};
  cuopt_float_t rhs[]          = {2.0};

  // Variable bounds: x1, x2 free; x3 (the cone head) >= 0
  cuopt_float_t var_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY, 0.0};
  cuopt_float_t var_upper_bounds[] = {CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY};
  char variable_types[]            = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  // Second-order cone constraint x1^2 + x2^2 - x3^2 <= 0, supplied as the
  // quadratic matrix Q in coordinate (triplet) form. There is no linear part
  // and the right-hand side must be 0.
  cuopt_int_t q_row_index[]  = {0, 1, 2};
  cuopt_int_t q_col_index[]  = {0, 1, 2};
  cuopt_float_t q_coeff[]    = {1.0, 1.0, -1.0};

  cuopt_int_t status;
  cuopt_float_t time;
  cuopt_int_t termination_status;
  cuopt_float_t objective_value;
  cuopt_float_t* solution_values = NULL;

  printf("Creating and solving simple SOCP problem...\n");

  // Create the linear part of the problem
  status = cuOptCreateProblem(num_constraints,
                              num_variables,
                              CUOPT_MINIMIZE,
                              0.0,  // objective offset
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
    printf("Error creating problem: %d\n", status);
    goto DONE;
  }

  // Add the second-order cone constraint (no linear term, rhs = 0)
  status = cuOptAddQuadraticConstraint(problem,
                                       3,  // number of quadratic entries
                                       q_row_index,
                                       q_col_index,
                                       q_coeff,
                                       0,     // number of linear entries
                                       NULL,  // linear indices
                                       NULL,  // linear coefficients
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

  status = cuOptSolve(problem, settings, &solution);
  if (status != CUOPT_SUCCESS) {
    printf("Error solving problem: %d\n", status);
    goto DONE;
  }

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

  status = cuOptGetObjectiveValue(solution, &objective_value);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

  printf("\nResults:\n");
  printf("--------\n");
  printf("Termination status: %s (%d)\n",
         termination_status_to_string(termination_status),
         termination_status);
  printf("Solve time: %f seconds\n", time);
  printf("Objective value: %f\n", objective_value);

  solution_values = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  if (solution_values == NULL) {
    printf("Error allocating solution values\n");
    goto DONE;
  }
  status = cuOptGetPrimalSolution(solution, solution_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solution values: %d\n", status);
    free(solution_values);
    goto DONE;
  }

  printf("\nPrimal Solution: Solution variables \n");
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

int main()
{
  cuopt_int_t status = test_simple_socp();

  if (status == CUOPT_SUCCESS) {
    printf("\nTest completed successfully!\n");
    return 0;
  } else {
    printf("\nTest failed with status: %d\n", status);
    return 1;
  }
}
