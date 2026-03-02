/*
 * Simple LP (C API): minimize -0.2*x1 + 0.1*x2
 * subject to 3*x1 + 4*x2 <= 5.4, 2.7*x1 + 10.1*x2 <= 4.9, x1,x2 >= 0
 */
#include <cuopt/linear_programming/cuopt_c.h>
#include <cuopt/linear_programming/constants.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    cuOptOptimizationProblem problem = NULL;
    cuOptSolverSettings settings = NULL;
    cuOptSolution solution = NULL;

    cuopt_int_t num_variables = 2;
    cuopt_int_t num_constraints = 2;

    cuopt_int_t row_offsets[] = {0, 2, 4};
    cuopt_int_t column_indices[] = {0, 1, 0, 1};
    cuopt_float_t values[] = {3.0, 4.0, 2.7, 10.1};

    cuopt_float_t objective_coefficients[] = {-0.2, 0.1};
    cuopt_float_t constraint_upper_bounds[] = {5.4, 4.9};
    cuopt_float_t constraint_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY};

    cuopt_float_t var_lower_bounds[] = {0.0, 0.0};
    cuopt_float_t var_upper_bounds[] = {CUOPT_INFINITY, CUOPT_INFINITY};
    char variable_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

    cuopt_int_t status = cuOptCreateRangedProblem(
        num_constraints, num_variables, CUOPT_MINIMIZE, 0.0,
        objective_coefficients,
        row_offsets, column_indices, values,
        constraint_lower_bounds, constraint_upper_bounds,
        var_lower_bounds, var_upper_bounds,
        variable_types, &problem
    );
    if (status != CUOPT_SUCCESS) {
        printf("Error creating problem: %d\n", status);
        return 1;
    }

    cuOptCreateSolverSettings(&settings);
    cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 0.0001);
    cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, 60.0);

    status = cuOptSolve(problem, settings, &solution);
    if (status != CUOPT_SUCCESS) {
        printf("Error solving: %d\n", status);
        goto cleanup;
    }

    cuopt_float_t time, objective_value;
    cuopt_int_t termination_status;
    cuOptGetSolveTime(solution, &time);
    cuOptGetTerminationStatus(solution, &termination_status);
    cuOptGetObjectiveValue(solution, &objective_value);

    printf("Status: %d\n", termination_status);
    printf("Time: %f s\n", time);
    printf("Objective: %f\n", objective_value);

    cuopt_float_t *sol = malloc((size_t)num_variables * sizeof(cuopt_float_t));
    if (sol) {
        cuOptGetPrimalSolution(solution, sol);
        printf("x1 = %f, x2 = %f\n", sol[0], sol[1]);
        free(sol);
    }

cleanup:
    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroySolution(&solution);
    return (status == CUOPT_SUCCESS) ? 0 : 1;
}
