/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Test script for cuOpt C API - works with both local and remote solve.
 *
 * Usage:
 *   # Local solve (default):
 *   ./test_c_api /path/to/problem.mps
 *
 *   # Remote solve (set environment variables first):
 *   CUOPT_REMOTE_HOST=localhost CUOPT_REMOTE_PORT=9090 ./test_c_api /path/to/problem.mps
 *
 * Build:
 *   gcc -I $CONDA_PREFIX/include -L $CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib \
 *       -o test_c_api test_c_api.c -lcuopt
 *
 * Example:
 *   ./test_c_api /home/tmckay/repos/HiGHS/check/instances/afiro.mps
 */

#include <cuopt/linear_programming/cuopt_c.h>
#include <stdio.h>
#include <stdlib.h>

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

cuopt_int_t solve_mps_file(const char* filename)
{
  cuOptOptimizationProblem problem = NULL;
  cuOptSolverSettings settings = NULL;
  cuOptSolution solution = NULL;
  cuopt_int_t status;
  cuopt_float_t time;
  cuopt_int_t termination_status;
  cuopt_float_t objective_value;
  cuopt_int_t num_variables;
  cuopt_float_t* solution_values = NULL;

  // Check for remote solve configuration
  const char* remote_host = getenv("CUOPT_REMOTE_HOST");
  const char* remote_port = getenv("CUOPT_REMOTE_PORT");
  if (remote_host && remote_port) {
    printf("Remote solve enabled: %s:%s\n", remote_host, remote_port);
  } else {
    printf("Local solve (no CUOPT_REMOTE_HOST/PORT set)\n");
  }

  printf("Reading MPS file: %s\n", filename);

  // Create the problem from MPS file
  status = cuOptReadProblem(filename, &problem);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating problem from MPS file: %d\n", status);
    goto DONE;
  }

  // Get problem size
  status = cuOptGetNumVariables(problem, &num_variables);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting number of variables: %d\n", status);
    goto DONE;
  }

  // Create solver settings
  status = cuOptCreateSolverSettings(&settings);
  if (status != CUOPT_SUCCESS) {
    printf("Error creating solver settings: %d\n", status);
    goto DONE;
  }

  // Set solver parameters
  status = cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 1e-6);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting optimality tolerance: %d\n", status);
    goto DONE;
  }

  // Solve the problem
  printf("Solving...\n");
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
  printf("----------------------------------------\n");
  printf("Number of variables: %d\n", num_variables);
  printf("Status: %s (%d)\n", termination_status_to_string(termination_status), termination_status);
  printf("Objective: %e\n", objective_value);
  printf("Solve time: %.3f seconds\n", time);

  // Get and print solution variables
  solution_values = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  status = cuOptGetPrimalSolution(solution, solution_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solution values: %d\n", status);
    goto DONE;
  }

  printf("\nPrimal solution (first 10 of %d variables):\n", num_variables);
  for (cuopt_int_t i = 0; i < (num_variables < 10 ? num_variables : 10); i++) {
    printf("  x%d = %f\n", i + 1, solution_values[i]);
  }
  if (num_variables > 10) {
    printf("  ... (%d more variables)\n", num_variables - 10);
  }

  printf("\nDone!\n");

DONE:
  free(solution_values);
  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);

  return status;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <mps_file_path>\n", argv[0]);
    printf("\nSet CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT for remote solve.\n");
    return 1;
  }

  cuopt_int_t status = solve_mps_file(argv[1]);

  if (status == CUOPT_SUCCESS) {
    return 0;
  } else {
    printf("\nSolver failed with status: %d\n", status);
    return 1;
  }
}
