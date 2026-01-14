/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Simple QP C API Example
 *
 * This example demonstrates how to use the cuOpt C API for quadratic programming.
 *
 * Problem:
 *   Minimize: x^2 + y^2
 *   Subject to:
 *     x + y >= 1
 *     x, y >= 0
 *
 *
 * Build:
 *   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o simple_qp_example simple_qp_example.c -lcuopt
 *
 * Run:
 *   ./simple_qp_example
 */

// Include the cuOpt linear programming solver header
#include <cuopt/linear_programming/cuopt_c.h>
#include <stdio.h>
#include <stdlib.h>

// Convert termination status to string
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
    default:
      return "Unknown";
  }
}

// Test simple QP problem
cuopt_int_t test_simple_qp()
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings = NULL;
  cuOptSolution solution = NULL;

  /* Solve the following QP:
     minimize x^2 + y^2
     subject to:
     x + y >= 1
     x, y >= 0
  */

  cuopt_int_t num_variables = 2;
  cuopt_int_t num_constraints = 1;
  cuopt_int_t nnz = 2;

  // CSR format constraint matrix
  // https://docs.nvidia.com/nvpl/latest/sparse/storage_format/sparse_matrix.html#compressed-sparse-row-csr
  cuopt_int_t row_offsets[] = {0, 2};
  cuopt_int_t column_indices[] = {0, 1};
  cuopt_float_t values[] = {1.0, 1.0};

  // Objective coefficients
  // From the objective function: minimize x^2 + y^2
  // 0 is the coefficient of the linear term on x
  // 0 is the coefficient of the linear term on y
  cuopt_float_t linear_objective_coefficients[] = {0.0, 0.0};

  // Quadratic objective matrix
  // From the objective function: minimize x^2 + y^2
  // 1 is the coefficient of the quadratic term on x^2
  // 1 is the coefficient of the quadratic term on y^2
  cuopt_float_t quadratic_objective_matrix_values[] = {1.0, 1.0};
  cuopt_int_t quadratic_objective_matrix_row_offsets[] = {0, 1, 2};
  cuopt_int_t quadratic_objective_matrix_column_indices[] = {0, 1};

  // Constraint bounds
  // From the constraints:
  // x + y >= 1
  cuopt_float_t constraint_rhs[] = {1.0};
  char constraint_sense[] = { CUOPT_GREATER_THAN };


  // Variable bounds
  // From the constraints:
  // x1, x2 >= 0
  cuopt_float_t var_lower_bounds[] = {0.0, 0.0};
  cuopt_float_t var_upper_bounds[] = {CUOPT_INFINITY, CUOPT_INFINITY};

  // Variable types (continuous)
  // From the constraints:
  // x1, x2 >= 0
  char variable_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

  cuopt_int_t status;
  cuopt_float_t time;
  cuopt_int_t termination_status;
  cuopt_float_t objective_value;

  printf("Creating and solving simple QP problem...\n");

  // Create the problem
  status = cuOptCreateQuadraticProblem(num_constraints,
                                       num_variables,
                                       CUOPT_MINIMIZE,
                                       0.0,  // objective offset
                                       linear_objective_coefficients,
                                       quadratic_objective_matrix_row_offsets,
                                       quadratic_objective_matrix_column_indices,
                                       quadratic_objective_matrix_values,
                                       row_offsets,
                                       column_indices,
                                       values,
                                       constraint_sense,
                                       constraint_rhs,
                                       var_lower_bounds,
                                       var_upper_bounds,
                                       variable_types,
                                       &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating problem: %d\n", status);
    goto DONE;
  }

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

  status = cuOptGetObjectiveValue(solution, &objective_value);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting objective value: %d\n", status);
    goto DONE;
  }

  // Print results
  printf("\nResults:\n");
  printf("--------\n");
  printf("Termination status: %s (%d)\n", termination_status_to_string(termination_status), termination_status);
  printf("Solve time: %f seconds\n", time);
  printf("Objective value: %f\n", objective_value);

  // Get and print solution variables
  cuopt_float_t* solution_values = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
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

int main() {
  // Run the test
  cuopt_int_t status = test_simple_qp();

  if (status == CUOPT_SUCCESS) {
    printf("\nTest completed successfully!\n");
    return 0;
  } else {
    printf("\nTest failed with status: %d\n", status);
    return 1;
  }
}
