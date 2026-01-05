LP C API Examples
=================


Example With Data
-----------------

This example demonstrates how to use the LP solver in C. More details on the API can be found in :doc:`C API <lp-milp-c-api>`.

The example code is available at ``examples/cuopt-c/lp/simple_lp_example.c`` (:download:`download <examples/simple_lp_example.c>`):

.. literalinclude:: examples/simple_lp_example.c
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
   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o simple_lp_example simple_lp_example.c -lcuopt
   ./simple_lp_example



You should see the following output:

.. code-block:: bash
   :caption: Output

   Creating and solving simple LP problem...
   Solving a problem with 2 constraints 2 variables (0 integers) and 4 nonzeros
   Objective offset 0.000000 scaling_factor 1.000000
   Running concurrent

   Dual simplex finished in 0.00 seconds
      Iter    Primal Obj.      Dual Obj.    Gap        Primal Res.  Dual Res.   Time
         0 +0.00000000e+00 +0.00000000e+00  0.00e+00   0.00e+00     2.00e-01   0.011s
   PDLP finished
   Concurrent time:  0.013s
   Solved with dual simplex
   Status: Optimal   Objective: -3.60000000e-01  Iterations: 1  Time: 0.013s

   Results:
   --------
   Termination status: Optimal (1)
   Solve time: 0.000013 seconds
   Objective value: -0.360000

   Primal Solution: Solution variables
   x1 = 1.800000
   x2 = 0.000000

   Test completed successfully!


Example With MPS File
---------------------

This example demonstrates how to use the cuOpt linear programming solver in C to solve an MPS file.

The example code is available at ``examples/cuopt-c/lp/mps_file_example.c`` (:download:`download <examples/mps_file_example.c>`):

.. literalinclude:: examples/mps_file_example.c
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

A sample MPS file (:download:`download sample.mps <examples/sample.mps>`):

.. literalinclude:: examples/sample.mps
   :language: text
   :linenos:

Build and run the example

.. code-block:: bash

   # Build and run the example
   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o mps_file_example mps_file_example.c -lcuopt
   ./mps_file_example sample.mps


You should see the following output:

.. code-block:: bash
   :caption: Output

   Reading and solving MPS file: sample.mps
   Solving a problem with 2 constraints 2 variables (0 integers) and 4 nonzeros
   Objective offset 0.000000 scaling_factor 1.000000
   Running concurrent

   Dual simplex finished in 0.00 seconds
      Iter    Primal Obj.      Dual Obj.    Gap        Primal Res.  Dual Res.   Time
         0 +0.00000000e+00 +0.00000000e+00  0.00e+00   0.00e+00     2.00e-01   0.012s
   PDLP finished
   Concurrent time:  0.014s
   Solved with dual simplex
   Status: Optimal   Objective: -3.60000000e-01  Iterations: 1  Time: 0.014s

   Results:
   --------
   Number of variables: 2
   Termination status: Optimal (1)
   Solve time: 0.000014 seconds
   Objective value: -0.360000

   Primal Solution: First 10 solution variables (or fewer if less exist):
   x1 = 1.800000
   x2 = 0.000000

   Solver completed successfully!


.. _simple-qp-example-c:

Simple Quadratic Programming Example
------------------------------------

.. note::
   The QP solver is currently in beta.

This example demonstrates how to use the cuOpt C API for quadratic programming.

The example code is available at ``examples/cuopt-c/lp/simple_qp_example.c`` (:download:`download <examples/simple_qp_example.c>`):

.. literalinclude:: examples/simple_qp_example.c
   :language: c
   :linenos:

Build and run the example

.. code-block:: bash

   # Build and run the example
   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o simple_qp_example simple_qp_example.c -lcuopt
   ./simple_qp_example

You should see the following output:

.. code-block:: bash
   :caption: Output

   Creating and solving simple QP problem...
   Status: Optimal
   Objective value: 0.500000
   x = 0.500000
   y = 0.500000
   Test completed successfully!
