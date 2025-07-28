MILP C API Examples
===================


Example With Data
-----------------

This example demonstrates how to use the MILP solver in C. More details on the API can be found in :doc:`C API <lp-milp-c-api>`.

Copy the code below into a file called ``milp_example.c``:

.. code-block:: c

   /*
    * Simple test program for cuOpt MILP solver
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

   // Test simple MILP problem
   cuopt_int_t test_simple_milp()
   {
     cuOptOptimizationProblem problem = NULL;
     cuOptSolverSettings settings = NULL;
     cuOptSolution solution = NULL;

     /* Solve the following LP:
        minimize -0.2*x1 + 0.1*x2
        subject to:
        3.0*x1 + 4.0*x2 <= 5.4
        2.7*x1 + 10.1*x2 <= 4.9
        x1, x2 >= 0
        x1 is integer
        x2 is continuous
     */

     cuopt_int_t num_variables = 2;
     cuopt_int_t num_constraints = 2;
     cuopt_int_t nnz = 4;

     // CSR format constraint matrix
     // https://docs.nvidia.com/nvpl/latest/sparse/storage_format/sparse_matrix.html#compressed-sparse-row-csr
     // From the constraints:
     // 3.0*x1 + 4.0*x2 <= 5.4
     // 2.7*x1 + 10.1*x2 <= 4.9
     cuopt_int_t row_offsets[] = {0, 2, 4};
     cuopt_int_t column_indices[] = {0, 1, 0, 1};
     cuopt_float_t values[] = {3.0, 4.0, 2.7, 10.1};

     // Objective coefficients
     // From the objective function: minimize 0.2*x1 + 0.1*x2
     // -0.2 is the coefficient of x1
     // 0.1 is the coefficient of x2
     cuopt_float_t objective_coefficients[] = {-0.2, 0.1};

     // Constraint bounds
     // From the constraints:
     // 3.0*x1 + 4.0*x2 <= 5.4
     // 2.7*x1 + 10.1*x2 <= 4.9
     cuopt_float_t constraint_upper_bounds[] = {5.4, 4.9};
     cuopt_float_t constraint_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY};

     // Variable bounds
     // From the constraints:
     // x1, x2 >= 0
     cuopt_float_t var_lower_bounds[] = {0.0, 0.0};
     cuopt_float_t var_upper_bounds[] = {CUOPT_INFINITY, CUOPT_INFINITY};

     // Variable types (continuous)
     // From the constraints:
     // x1, x2 >= 0
     // x1 is integer
     // x2 is continuous
     char variable_types[] = {CUOPT_INTEGER, CUOPT_CONTINUOUS};

     cuopt_int_t status;
     cuopt_float_t time;
     cuopt_int_t termination_status;
     cuopt_float_t objective_value;

     printf("Creating and solving simple LP problem...\n");

     // Create the problem
     status = cuOptCreateRangedProblem(num_constraints,
                                      num_variables,
                                      CUOPT_MINIMIZE,  // minimize=False
                                      0.0,            // objective offset
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

     // Set solver parameters
     status = cuOptSetFloatParameter(settings, CUOPT_MIP_ABSOLUTE_TOLERANCE, 0.0001);
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
     printf("Termination status: %s (%d)\n", termination_status_to_string(termination_status), termination_status);
     printf("Solve time: %f seconds\n", time);
     printf("Objective value: %f\n", objective_value);

     // Get and print solution variables
     cuopt_float_t* solution_values = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
     status = cuOptGetPrimalSolution(solution, solution_values);
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

   int main() {
     // Run the test
     cuopt_int_t status = test_simple_milp();

     if (status == CUOPT_SUCCESS) {
       printf("\nTest completed successfully!\n");
       return 0;
     } else {
       printf("\nTest failed with status: %d\n", status);
       return 1;
     }
   }


It is necessary to have the path for include and library dirs ready, if you know the paths, please add them to the path variables directly. Otherwise, run the following commands to find the path and assign it to the path variables.
The following commands are for Linux and might fail in cases where the cuopt library is not installed or there are multiple cuopt libraries in the system.

If you have built it locally, libcuopt.so will be in the build directory ``cpp/build`` and include directoy would be ``cpp/include``.

.. code-block:: bash

   # Find the cuopt header file and assign to INCLUDE_PATH
   INCLUDE_PATH=$(find / -name "cuopt_c.h" -path "*/linear_programming/*" -printf "%h\n" | sed 's/\/linear_programming//' 2>/dev/null)
   # Find the libcuopt library and assign to LIBCUOPT_LIBRARY_PATH
   LIBCUOPT_LIBRARY_PATH=$(find / -name "libcuopt.so" 2>/dev/null)


Build and run the example

.. code-block:: bash

   # Build and run the example
   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o milp_example milp_example.c -lcuopt
   ./milp_example



You should see the following output:

.. code-block:: bash
  :caption: Output

   Creating and solving simple LP problem...
   Solving a problem with 2 constraints 2 variables (1 integers) and 4 nonzeros
   Objective offset 0.000000 scaling_factor 1.000000
   After trivial presolve updated 2 constraints 2 variables
   Running presolve!
   After trivial presolve updated 2 constraints 2 variables
   Solving LP root relaxation
   Scaling matrix. Maximum column norm 1.046542e+00
   Dual Simplex Phase 1
   Dual feasible solution found.
   Dual Simplex Phase 2
    Iter     Objective   Primal Infeas  Perturb  Time
       1 -2.00000000e-01 1.46434160e+00 0.00e+00 0.00

   Root relaxation solution found in 2 iterations and 0.00s
   Root relaxation objective -2.00000000e-01

   Optimal solution found at root node. Objective -2.0000000000000001e-01. Time 0.00.
   B&B added a solution to population, solution queue size 0 with objective -0.2
   Solution objective: -0.200000 , relative_mip_gap 0.000000 solution_bound -0.200000 presolve_time 0.041144 total_solve_time 0.000000 max constraint violation 0.000000 max int violation 0.000000 max var bounds violation 0.000000 nodes 0 simplex_iterations 0

   Results:
   --------
   Termination status: Optimal (1)
   Solve time: 0.000000 seconds
   Objective value: -0.200000

   Solution:
   x1 = 1.000000
   x2 = 0.000000

   Test completed successfully!


Example With MPS File
---------------------

This example demonstrates how to use the cuOpt solver in C to solve an MPS file.

Copy the code below into a file called ``milp_example_mps.c``:

.. code-block:: c

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


It is necessary to have the path for include and library dirs ready, if you know the paths, please add them to the path variables directly. Otherwise, run the following commands to find the path and assign it to the path variables.
The following commands are for Linux and might fail in cases where the cuopt library is not installed or there are multiple cuopt libraries in the system.

If you have built it locally, libcuopt.so will be in the build directory ``cpp/build`` and include directoy would be ``cpp/include``.

.. code-block:: bash

   # Find the cuopt header file and assign to INCLUDE_PATH
   INCLUDE_PATH=$(find / -name "cuopt_c.h" -path "*/linear_programming/*" -printf "%h\n" | sed 's/\/linear_programming//' 2>/dev/null)
   # Find the libcuopt library and assign to LIBCUOPT_LIBRARY_PATH
   LIBCUOPT_LIBRARY_PATH=$(find / -name "libcuopt.so" 2>/dev/null)

Build and run the example

.. code-block:: bash

    # Create a MPS file in the current directory
    echo "* Example 2.1 from N & W
   * Optimal solution -28
   NAME          EXAMPLE21
   ROWS
    N  OBJ
    L  C1
    L  C2
    L  C3
   COLUMNS
     MARK0001  'MARKER'                 'INTORG'
       X1        OBJ             -7
       X1        C1              -1
       X1        C2               5
       X1        C3              -2
       X2        OBJ             -2
       X2        C1               2
       X2        C2               1
       X2        C3              -2
     MARK0001  'MARKER'                 'INTEND'
   RHS
       RHS       C1               4
       RHS       C2              20
       RHS       C3              -7
   BOUNDS
    UP BOUND     X1               10
    UP BOUND     X2               10
   ENDATA" > sample.mps

   # Build and run the example
   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o milp_example_mps milp_example_mps.c -lcuopt
   ./milp_example_mps sample.mps


You should see the following output:

.. code-block:: bash
  :caption: Output

   Reading and solving MPS file: sample.mps
   Solving a problem with 3 constraints 2 variables (2 integers) and 6 nonzeros
   Objective offset 0.000000 scaling_factor 1.000000
   After trivial presolve updated 3 constraints 2 variables
   Running presolve!
   After trivial presolve updated 3 constraints 2 variables
   Solving LP root relaxation
   Scaling matrix. Maximum column norm 1.225464e+00
   Dual Simplex Phase 1
   Dual feasible solution found.
   Dual Simplex Phase 2
    Iter     Objective   Primal Infeas  Perturb  Time
       1 -3.04000000e+01 7.57868205e+00 0.00e+00 0.00

   Root relaxation solution found in 3 iterations and 0.00s
   Root relaxation objective -3.01818182e+01

   Strong branching on 2 fractional variables
   | Explored | Unexplored | Objective   |    Bound    | Depth | Iter/Node |  Gap   |    Time
           0        1                +inf  -3.018182e+01      1   0.0e+00       -        0.00
   B       3        1       -2.700000e+01  -2.980000e+01      2   6.7e-01     10.4%      0.00
   B&B added a solution to population, solution queue size 0 with objective -27
   B       4        0       -2.800000e+01  -2.980000e+01      2   7.5e-01      6.4%      0.00
   B&B added a solution to population, solution queue size 1 with objective -28
   Explored 4 nodes in 0.00s.
   Absolute Gap 0.000000e+00 Objective -2.8000000000000004e+01 Lower Bound -2.8000000000000004e+01
   Optimal solution found.
   Generated fast solution in 0.136067 seconds with objective -28.000000
   Solution objective: -28.000000 , relative_mip_gap 0.000000 solution_bound -28.000000 presolve_time 0.039433 total_solve_time 0.000000 max constraint violation 0.000000 max int violation 0.000000 max var bounds violation 0.000000 nodes 4 simplex_iterations 3

   Results:
   --------
   Number of variables: 2
   Termination status: Optimal (1)
   Solve time: 0.000000 seconds
   Objective value: -28.000000

   Solution:
   x1 = 4.000000
   x2 = 0.000000

   Solver completed successfully!
