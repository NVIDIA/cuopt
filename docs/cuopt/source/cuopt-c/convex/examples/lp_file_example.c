/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * LP File C API Example
 *
 * This example demonstrates how to solve an LP problem from an LP format
 * file using the cuOpt C API. The same ``cuOptReadProblem`` call handles
 * both MPS and LP inputs — the format is dispatched automatically by the
 * file extension (case-insensitive): ``.lp`` / ``.lp.gz`` / ``.lp.bz2``
 * go to the LP parser; ``.mps`` / ``.qps`` and their ``.gz`` / ``.bz2``
 * variants go to the MPS parser.
 *
 * Problem (from LP file):
 *   Minimize: -0.2*VAR1 + 0.1*VAR2
 *   Subject to:
 *     3*VAR1 + 4*VAR2 <= 5.4
 *     2.7*VAR1 + 10.1*VAR2 <= 4.9
 *     VAR1, VAR2 >= 0
 *
 * Expected Output:
 *   Number of variables: 2
 *   Termination status: Optimal (1)
 *   Objective value: -0.360000
 *   x1 = 1.800000
 *   x2 = 0.000000
 *
 * Build:
 *   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o lp_file_example lp_file_example.c -lcuopt
 *
 * Run:
 *   ./lp_file_example sample.lp
 */

#include <cuopt/mathematical_optimization/cuopt_c.h>
#include <stdio.h>
#include <stdlib.h>

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
    default:
      return "Unknown";
  }
}

cuopt_int_t solve_lp_file(const char* filename)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings     = NULL;
  cuOptSolution solution           = NULL;
  cuopt_int_t status;
  cuopt_float_t time;
  cuopt_int_t termination_status;
  cuopt_float_t objective_value;
  cuopt_int_t num_variables;
  cuopt_float_t* solution_values = NULL;

  printf("Reading and solving input file: %s\n", filename);

  // Create the problem from the input file. cuOptReadProblem dispatches on
  // the file extension (case-insensitive): ``.lp`` / ``.lp.gz`` / ``.lp.bz2``
  // go to the LP parser; ``.mps`` / ``.qps`` and their ``.gz`` / ``.bz2``
  // variants go to the MPS parser.
  status = cuOptReadProblem(filename, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating problem from input file: %d\n", status);
    goto DONE;
  }

  status = cuOptGetNumVariables(problem, &num_variables);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting number of variables: %d\n", status);
    goto DONE;
  }

  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  status = cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 0.0001);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting optimality tolerance: %d\n", status);
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
  printf("Number of variables: %d\n", num_variables);
  printf("Termination status: %s (%d)\n",
         termination_status_to_string(termination_status),
         termination_status);
  printf("Solve time: %f seconds\n", time);
  printf("Objective value: %f\n", objective_value);

  solution_values = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  status          = cuOptGetPrimalSolution(solution, solution_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solution values: %d\n", status);
    goto DONE;
  }

  printf("\nPrimal Solution: First 10 solution variables (or fewer if less exist):\n");
  for (cuopt_int_t i = 0; i < (num_variables < 10 ? num_variables : 10); i++) {
    printf("x%d = %f\n", i + 1, solution_values[i]);
  }
  if (num_variables > 10) {
    printf("... (showing only first 10 of %d variables)\n", num_variables);
  }

DONE:
  free(solution_values);
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <lp_or_mps_file_path>\n", argv[0]);
    return 1;
  }

  cuopt_int_t status = solve_lp_file(argv[1]);

  if (status == CUOPT_SUCCESS) {
    printf("\nSolver completed successfully!\n");
    return 0;
  } else {
    printf("\nSolver failed with status: %d\n", status);
    return 1;
  }
}
