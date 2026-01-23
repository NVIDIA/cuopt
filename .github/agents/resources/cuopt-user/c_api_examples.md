# C API examples (cuOpt)

## C API: Simple LP Example

```c
/*
 * Simple LP C API Example
 *
 * Solve: minimize  -0.2*x1 + 0.1*x2
 *        subject to  3.0*x1 + 4.0*x2 <= 5.4
 *                    2.7*x1 + 10.1*x2 <= 4.9
 *                    x1, x2 >= 0
 *
 * Expected: x1 = 1.8, x2 = 0.0, objective = -0.36
 */
#include <cuopt/linear_programming/cuopt_c.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cuOptOptimizationProblem problem = NULL;
    cuOptSolverSettings settings = NULL;
    cuOptSolution solution = NULL;

    cuopt_int_t num_variables = 2;
    cuopt_int_t num_constraints = 2;

    // Constraint matrix in CSR format
    cuopt_int_t row_offsets[] = {0, 2, 4};
    cuopt_int_t column_indices[] = {0, 1, 0, 1};
    cuopt_float_t values[] = {3.0, 4.0, 2.7, 10.1};

    // Objective coefficients: minimize -0.2*x1 + 0.1*x2
    cuopt_float_t objective_coefficients[] = {-0.2, 0.1};

    // Constraint bounds (ranged form: lower <= Ax <= upper)
    cuopt_float_t constraint_upper_bounds[] = {5.4, 4.9};
    cuopt_float_t constraint_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY};

    // Variable bounds: x1, x2 >= 0
    cuopt_float_t var_lower_bounds[] = {0.0, 0.0};
    cuopt_float_t var_upper_bounds[] = {CUOPT_INFINITY, CUOPT_INFINITY};

    // Variable types: both continuous
    char variable_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

    cuopt_int_t status;
    cuopt_float_t time;
    cuopt_int_t termination_status;
    cuopt_float_t objective_value;

    // Create the problem
    status = cuOptCreateRangedProblem(
        num_constraints,
        num_variables,
        CUOPT_MINIMIZE,
        0.0,                      // objective offset
        objective_coefficients,
        row_offsets,
        column_indices,
        values,
        constraint_lower_bounds,
        constraint_upper_bounds,
        var_lower_bounds,
        var_upper_bounds,
        variable_types,
        &problem
    );
    if (status != CUOPT_SUCCESS) {
        printf("Error creating problem: %d\n", status);
        return 1;
    }

    // Create solver settings
    status = cuOptCreateSolverSettings(&settings);
    if (status != CUOPT_SUCCESS) {
        printf("Error creating solver settings: %d\n", status);
        goto DONE;
    }

    // Set solver parameters
    cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 0.0001);
    cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, 60.0);

    // Solve the problem
    status = cuOptSolve(problem, settings, &solution);
    if (status != CUOPT_SUCCESS) {
        printf("Error solving problem: %d\n", status);
        goto DONE;
    }

    // Get and print results
    cuOptGetSolveTime(solution, &time);
    cuOptGetTerminationStatus(solution, &termination_status);
    cuOptGetObjectiveValue(solution, &objective_value);

    printf("Termination status: %d\n", termination_status);
    printf("Solve time: %f seconds\n", time);
    printf("Objective value: %f\n", objective_value);

    // Get solution values
    cuopt_float_t* solution_values = (cuopt_float_t*)malloc(
        num_variables * sizeof(cuopt_float_t)
    );
    cuOptGetPrimalSolution(solution, solution_values);
    for (cuopt_int_t i = 0; i < num_variables; i++) {
        printf("x%d = %f\n", i + 1, solution_values[i]);
    }
    free(solution_values);

DONE:
    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroySolution(&solution);

    return (status == CUOPT_SUCCESS) ? 0 : 1;
}
```

## C API: MILP Example (with integer variables)

```c
/*
 * Simple MILP C API Example
 *
 * Solve: minimize  -0.2*x1 + 0.1*x2
 *        subject to  3.0*x1 + 4.0*x2 <= 5.4
 *                    2.7*x1 + 10.1*x2 <= 4.9
 *                    x1 integer, x2 continuous, both >= 0
 */
#include <cuopt/linear_programming/cuopt_c.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cuOptOptimizationProblem problem = NULL;
    cuOptSolverSettings settings = NULL;
    cuOptSolution solution = NULL;

    cuopt_int_t num_variables = 2;
    cuopt_int_t num_constraints = 2;

    // Constraint matrix in CSR format
    cuopt_int_t row_offsets[] = {0, 2, 4};
    cuopt_int_t column_indices[] = {0, 1, 0, 1};
    cuopt_float_t values[] = {3.0, 4.0, 2.7, 10.1};

    // Objective coefficients
    cuopt_float_t objective_coefficients[] = {-0.2, 0.1};

    // Constraint bounds
    cuopt_float_t constraint_upper_bounds[] = {5.4, 4.9};
    cuopt_float_t constraint_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY};

    // Variable bounds
    cuopt_float_t var_lower_bounds[] = {0.0, 0.0};
    cuopt_float_t var_upper_bounds[] = {CUOPT_INFINITY, CUOPT_INFINITY};

    // Variable types: x1 = INTEGER, x2 = CONTINUOUS
    char variable_types[] = {CUOPT_INTEGER, CUOPT_CONTINUOUS};

    cuopt_int_t status;
    cuopt_float_t time;
    cuopt_int_t termination_status;
    cuopt_float_t objective_value;

    // Create the problem (same API, but with integer variable types)
    status = cuOptCreateRangedProblem(
        num_constraints,
        num_variables,
        CUOPT_MINIMIZE,
        0.0,
        objective_coefficients,
        row_offsets,
        column_indices,
        values,
        constraint_lower_bounds,
        constraint_upper_bounds,
        var_lower_bounds,
        var_upper_bounds,
        variable_types,
        &problem
    );
    if (status != CUOPT_SUCCESS) {
        printf("Error creating problem: %d\n", status);
        return 1;
    }

    // Create solver settings
    status = cuOptCreateSolverSettings(&settings);
    if (status != CUOPT_SUCCESS) goto DONE;

    // Set MIP-specific parameters
    cuOptSetFloatParameter(settings, CUOPT_MIP_ABSOLUTE_TOLERANCE, 0.0001);
    cuOptSetFloatParameter(settings, CUOPT_MIP_RELATIVE_GAP, 0.01);  // 1% gap
    cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, 120.0);

    // Solve
    status = cuOptSolve(problem, settings, &solution);
    if (status != CUOPT_SUCCESS) goto DONE;

    // Get results
    cuOptGetSolveTime(solution, &time);
    cuOptGetTerminationStatus(solution, &termination_status);
    cuOptGetObjectiveValue(solution, &objective_value);

    printf("Termination status: %d\n", termination_status);
    printf("Solve time: %f seconds\n", time);
    printf("Objective value: %f\n", objective_value);

    cuopt_float_t* solution_values = malloc(num_variables * sizeof(cuopt_float_t));
    cuOptGetPrimalSolution(solution, solution_values);
    printf("x1 (integer) = %f\n", solution_values[0]);
    printf("x2 (continuous) = %f\n", solution_values[1]);
    free(solution_values);

DONE:
    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroySolution(&solution);

    return (status == CUOPT_SUCCESS) ? 0 : 1;
}
```

## C API: Build & Run

```bash
# Find include and library paths (adjust based on installation)
# If installed via conda:
export INCLUDE_PATH="${CONDA_PREFIX}/include"
export LIBCUOPT_LIBRARY_PATH="${CONDA_PREFIX}/lib"

# Or find automatically:
# INCLUDE_PATH=$(find / -name "cuopt_c.h" -path "*/linear_programming/*" \
#                -printf "%h\n" | sed 's/\/linear_programming//' 2>/dev/null)
# LIBCUOPT_LIBRARY_PATH=$(dirname $(find / -name "libcuopt.so" 2>/dev/null))

# Compile
gcc -I ${INCLUDE_PATH} -L ${LIBCUOPT_LIBRARY_PATH} \
    -o simple_lp_example simple_lp_example.c -lcuopt

# Run
LD_LIBRARY_PATH=${LIBCUOPT_LIBRARY_PATH}:$LD_LIBRARY_PATH ./simple_lp_example
```
