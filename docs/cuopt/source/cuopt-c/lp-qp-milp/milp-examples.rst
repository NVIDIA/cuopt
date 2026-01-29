MILP C API Examples
===================


Example With Data
-----------------

This example demonstrates how to use the MILP solver in C. More details on the API can be found in :doc:`C API <lp-qp-milp-c-api>`.

The example code is available at ``../lp-milp/examples/simple_milp_example.c`` (:download:`download <examples/simple_milp_example.c>`):

.. literalinclude:: examples/simple_milp_example.c
   :language: c
   :linenos:

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
   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o simple_milp_example simple_milp_example.c -lcuopt
   ./simple_milp_example



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

The example code is available at ``examples/milp_mps_example.c`` (:download:`download <examples/milp_mps_example.c>`):

.. literalinclude:: examples/milp_mps_example.c
   :language: c
   :linenos:

It is necessary to have the path for include and library dirs ready, if you know the paths, please add them to the path variables directly. Otherwise, run the following commands to find the path and assign it to the path variables.
The following commands are for Linux and might fail in cases where the cuopt library is not installed or there are multiple cuopt libraries in the system.

If you have built it locally, libcuopt.so will be in the build directory ``cpp/build`` and include directoy would be ``cpp/include``.

.. code-block:: bash

   # Find the cuopt header file and assign to INCLUDE_PATH
   INCLUDE_PATH=$(find / -name "cuopt_c.h" -path "*/linear_programming/*" -printf "%h\n" | sed 's/\/linear_programming//' 2>/dev/null)
   # Find the libcuopt library and assign to LIBCUOPT_LIBRARY_PATH
   LIBCUOPT_LIBRARY_PATH=$(find / -name "libcuopt.so" 2>/dev/null)

A sample MILP MPS file (:download:`download mip_sample.mps <examples/mip_sample.mps>`):

.. literalinclude:: examples/mip_sample.mps
   :language: text
   :linenos:

Build and run the example

.. code-block:: bash

   # Build and run the example
   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o milp_mps_example milp_mps_example.c -lcuopt
   ./milp_mps_example mip_sample.mps


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
