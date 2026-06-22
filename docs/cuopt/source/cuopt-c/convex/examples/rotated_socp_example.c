/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
/*
 * Rotated SOCP C API Example
 *
 * Demonstrates a rotated second-order cone with the cuOpt C API. A linear
 * problem is created with cuOptCreateProblem, then a rotated cone is added with
 * cuOptAddQuadraticConstraint.
 *
 * The quadratic matrix Q must be supplied symmetrically: the cross term of a
 * rotated cone is given as the two equal off-diagonal entries
 * Q[x3, x4] = Q[x4, x3] = -0.5, so that x^T Q x contributes -x3*x4.
 *
 * Problem:
 *   minimize    x3 + x4
 *   subject to  x1 + x2 >= 2
 *               x1^2 + x2^2 - x3*x4 <= 0
 *               x3 >= 0, x4 >= 0,  x1, x2 free
 *
 * Optimal: x1 = x2 = 1, x3 = x4 = sqrt(2) ~= 1.414214, objective = 2*sqrt(2) ~= 2.828427.
 *
 * Build:
 *   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o rotated_socp_example rotated_socp_example.c -lcuopt
 *
 * Run:
 *   ./rotated_socp_example
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

cuopt_int_t test_rotated_socp()
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;

  cuopt_int_t num_variables   = 4;  // x1, x2, x3, x4
  cuopt_int_t num_constraints = 1;  // linear: x1 + x2 >= 2 (the cone is added separately)

  // Linear objective: minimize x3 + x4
  cuopt_float_t objective_coefficients[] = {0.0, 0.0, 1.0, 1.0};

  // Linear constraint matrix in CSR: row 0 is x1 + x2
  cuopt_int_t row_offsets[]    = {0, 2};
  cuopt_int_t column_indices[] = {0, 1};
  cuopt_float_t values[]       = {1.0, 1.0};
  char constraint_sense[]      = {CUOPT_GREATER_THAN};
  cuopt_float_t rhs[]          = {2.0};

  // Variable bounds: x1, x2 free; x3, x4 (the cone heads) >= 0
  cuopt_float_t var_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY, 0.0, 0.0};
  cuopt_float_t var_upper_bounds[] = {
    CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY, CUOPT_INFINITY};
  char variable_types[] = {
    CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  // Rotated cone x1^2 + x2^2 - x3*x4 <= 0, supplied as a symmetric quadratic
  // matrix Q in coordinate (triplet) form. The cross term is split into the two
  // equal off-diagonal entries Q[x3, x4] = Q[x4, x3] = -0.5. rhs must be 0.
  cuopt_int_t q_row_index[]  = {0, 1, 2, 3};
  cuopt_int_t q_col_index[]  = {0, 1, 3, 2};
  cuopt_float_t q_coeff[]    = {1.0, 1.0, -0.5, -0.5};

  cuopt_int_t status;
  cuopt_float_t time;
  cuopt_int_t termination_status;
  cuopt_float_t objective_value;
  cuopt_float_t* solution_values = NULL;

  printf("Creating and solving rotated SOCP problem...\n");

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

  // Add the rotated second-order cone constraint (no linear term, rhs = 0)
  status = cuOptAddQuadraticConstraint(problem,
                                       4,  // number of quadratic entries
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
  cuopt_int_t status = test_rotated_socp();

  if (status == CUOPT_SUCCESS) {
    printf("\nTest completed successfully!\n");
    return 0;
  } else {
    printf("\nTest failed with status: %d\n", status);
    return 1;
  }
}
