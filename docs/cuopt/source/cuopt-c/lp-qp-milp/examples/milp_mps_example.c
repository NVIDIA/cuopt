/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Example program for solving MPS files with cuOpt MILP solver
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

  printf("Reading and solving MPS file: %s\n", filename);

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
  status = cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 0.0001);
  if (status != CUOPT_SUCCESS) {
    printf("Error setting optimality tolerance: %d\n", status);
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
  printf("Number of variables: %d\n", num_variables);
  printf("Termination status: %s (%d)\n", termination_status_to_string(termination_status), termination_status);
  printf("Solve time: %f seconds\n", time);
  printf("Objective value: %f\n", objective_value);

  // Get and print solution variables
  solution_values = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
  status = cuOptGetPrimalSolution(solution, solution_values);
  if (status != CUOPT_SUCCESS) {
    printf("Error getting solution values: %d\n", status);
    goto DONE;
  }

  printf("\nSolution: \n");
  for (cuopt_int_t i = 0; i < num_variables; i++) {
    printf("x%d = %f\n", i + 1, solution_values[i]);
  }

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
    return 1;
  }

  // Run the solver
  cuopt_int_t status = solve_mps_file(argv[1]);

  if (status == CUOPT_SUCCESS) {
    printf("\nSolver completed successfully!\n");
    return 0;
  } else {
    printf("\nSolver failed with status: %d\n", status);
    return 1;
  }
}
