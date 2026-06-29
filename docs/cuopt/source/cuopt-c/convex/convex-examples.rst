==================================
Convex Optimization C API Examples
==================================


LP Example With Data
--------------------

This example demonstrates how to use the LP solver in C. More details on the API can be found in :doc:`C API <convex-c-api>`.

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


LP Example With MPS File
------------------------

This example demonstrates how to use the cuOpt linear programming solver in C to solve an MPS file.

The same ``cuOptReadProblem`` call also accepts **LP** format files. The
format is dispatched from the filename extension (case-insensitive):
``.lp`` / ``.lp.gz`` / ``.lp.bz2`` → LP parser; ``.mps`` / ``.qps`` and
their ``.gz`` / ``.bz2`` variants → MPS parser. Unknown extensions are
rejected. See :ref:`lp-file-example-c` for an LP counterpart.

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

A sample MPS file (:download:`download sample.mps <https://raw.githubusercontent.com/BUGSENG/PPL/devel/demos/ppl_lpsol/examples/sample.mps>`):

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


.. _lp-file-example-c:

LP Example With LP File
-----------------------

``cuOptReadProblem`` also accepts LP format files. The same function is
used — it dispatches on the file extension (case-insensitive):
``.lp`` / ``.lp.gz`` / ``.lp.bz2`` → LP parser; ``.mps`` / ``.qps`` and
their ``.gz`` / ``.bz2`` variants → MPS parser; unknown extensions are
rejected. See the ``read_lp`` declaration in
``cuopt/mathematical_optimization/io/parser.hpp`` for the supported subset of
the LP format.

The example code is available at ``examples/cuopt-c/lp/lp_file_example.c`` (:download:`download <examples/lp_file_example.c>`):

.. literalinclude:: examples/lp_file_example.c
   :language: c
   :linenos:

A sample LP file (:download:`download sample.lp <examples/sample.lp>`),
equivalent to the MPS sample above:

.. literalinclude:: examples/sample.lp
   :language: text
   :linenos:

Build and run the example

.. code-block:: bash

   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o lp_file_example lp_file_example.c -lcuopt
   ./lp_file_example sample.lp

The output matches the MPS example above (same problem, same objective = -0.36).


.. _simple-qp-example-c:

Simple Quadratic Programming Example
------------------------------------

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

   Results:
   --------
   Termination status: Optimal (1)
   Objective value: 0.500000

   Primal Solution: Solution variables
   x1 = 0.500000
   x2 = 0.500000

   Test completed successfully!


.. _simple-socp-example-c:

Simple Second-Order Cone Programming Example
--------------------------------------------

This example builds an SOCP directly with the C API: a linear problem is created with :c:func:`cuOptCreateProblem`, then a second-order cone constraint is added with :c:func:`cuOptAddQuadraticConstraint`. It minimizes ``x3`` subject to ``x1 + x2 >= 2`` and the cone ``||(x1, x2)||_2 <= x3``, written as the quadratic inequality ``x1^2 + x2^2 - x3^2 <= 0``. cuOpt detects the second-order cone structure and solves with the barrier method.

The example code is available at ``examples/cuopt-c/lp/simple_socp_example.c`` (:download:`download <examples/simple_socp_example.c>`):

.. literalinclude:: examples/simple_socp_example.c
   :language: c
   :linenos:

Build and run the example

.. code-block:: bash

   # Build and run the example
   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o simple_socp_example simple_socp_example.c -lcuopt
   ./simple_socp_example

The optimum is ``x1 = x2 = 1`` and ``x3 = sqrt(2)``:

.. code-block:: bash
   :caption: Output

   Creating and solving simple SOCP problem...

   Results:
   --------
   Termination status: Optimal (1)
   Objective value: 1.414214

   Primal Solution: Solution variables
   x1 = 1.000000
   x2 = 1.000000
   x3 = 1.414214

   Test completed successfully!

Note that dual variables are not currently returned for problems with quadratic constraints.


.. _rotated-socp-example-c:

Rotated Second-Order Cone Example
---------------------------------

This example adds a **rotated** cone with :c:func:`cuOptAddQuadraticConstraint`. The
quadratic matrix ``Q`` must be supplied **symmetrically**, so the cross term of
``x1^2 + x2^2 <= x3*x4`` is given as the two equal off-diagonal entries
``Q[x3, x4] = Q[x4, x3] = -0.5`` (triplets ``(2, 3, -0.5)`` and ``(3, 2, -0.5)``).
It minimizes ``x3 + x4`` subject to ``x1 + x2 >= 2`` and the rotated cone.

The example code is available at ``examples/cuopt-c/lp/rotated_socp_example.c`` (:download:`download <examples/rotated_socp_example.c>`):

.. literalinclude:: examples/rotated_socp_example.c
   :language: c
   :linenos:

Build and run the example

.. code-block:: bash

   # Build and run the example
   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o rotated_socp_example rotated_socp_example.c -lcuopt
   ./rotated_socp_example

The optimum is ``x1 = x2 = 1`` and ``x3 = x4 = sqrt(2)``:

.. code-block:: bash
   :caption: Output

   Creating and solving rotated SOCP problem...

   Results:
   --------
   Termination status: Optimal (1)
   Objective value: 2.828427

   Primal Solution: Solution variables
   x1 = 1.000000
   x2 = 1.000000
   x3 = 1.414214
   x4 = 1.414214

   Test completed successfully!

Note that dual variables are not currently returned for problems with quadratic constraints.


.. _general-quadratic-example-c:

General Convex Quadratic Constraint Example
-------------------------------------------

This example adds a general convex quadratic constraint with
:c:func:`cuOptAddQuadraticConstraint`. Here we add the convex quadratic constraint
``2*x^2 + 2*x*y + 2*y^2 <= 6``. Note that the quadratic matrix Q that encodes this
constraint need no be symmetric. Here the term ``2*x*y`` is supplied as a single entry ``Q[0,1] = 2``.

The example code is available at ``examples/cuopt-c/lp/general_quadratic_example.c`` (:download:`download <examples/general_quadratic_example.c>`):

.. literalinclude:: examples/general_quadratic_example.c
   :language: c
   :linenos:

Build and run the example

.. code-block:: bash

   # Build and run the example
   gcc -I $INCLUDE_PATH -L $LIBCUOPT_LIBRARY_PATH -o general_quadratic_example general_quadratic_example.c -lcuopt
   ./general_quadratic_example

The optimum is ``x = y = -1``:

.. code-block:: bash
   :caption: Output

   Creating and solving general convex quadratic problem...

   Results:
   --------
   Termination status: Optimal (1)
   Objective value: -2.000000

   Primal Solution: Solution variables
   x1 = -1.000000
   x2 = -1.000000

   Test completed successfully!


.. _qp-mps-example-c:

QP Example With MPS File
------------------------

The same ``mps_file_example.c`` program also solves quadratic-objective (QP)
problems. A quadratic objective is supplied through a ``QUADOBJ`` section, which
holds the (upper-triangular) entries of the objective matrix ``Q`` using the
MPS ``1/2 * x^T Q x`` convention.

A sample QP file (:download:`download qp_sample.mps <examples/qp_sample.mps>`),
which minimizes ``x^2 + y^2`` subject to ``x + y >= 1``:

.. literalinclude:: examples/qp_sample.mps
   :language: text
   :linenos:

Run it with the same binary:

.. code-block:: bash

   ./mps_file_example qp_sample.mps

cuOpt detects the quadratic objective and solves with the barrier method. The
optimum is ``x = y = 0.5`` with objective ``0.5``:

.. code-block:: bash
   :caption: Output (barrier iteration log abbreviated)

   Reading and solving MPS file: qp_sample.mps
   Problem has a quadratic objective. Solving with barrier.
   Barrier solver: 1 constraints, 3 variables, 3 nonzeros
   Quadratic objective matrix  : 2 nonzeros
   ...
   Optimal solution found in 11 iterations and 0.188s
   Objective +5.00000008e-01
   Barrier finished in 0.19 seconds

   Results:
   --------
   Number of variables: 2
   Termination status: Optimal (1)
   Solve time: 0.190449 seconds
   Objective value: 0.500000

   Primal Solution: First 10 solution variables (or fewer if less exist):
   x1 = 0.500000
   x2 = 0.500000

   Solver completed successfully!


.. _socp-mps-example-c:

SOCP Example With MPS File
--------------------------

Second-order cone (SOCP) constraints are expressed through ``QCMATRIX`` sections
— one per quadratic constraint — each holding the **full, symmetric** entries of
that constraint's quadratic matrix with a zero right-hand side. Both standard and
rotated cones are supported; see the SOCP section of
:doc:`LP/QP/QCQP/SOCP Features </convex-features>` for the cone forms.

A sample SOCP file (:download:`download socp_sample.mps <examples/socp_sample.mps>`)
minimizes ``s + p + q`` subject to ``a + b >= 2`` and two cones:

- standard cone ``||(a, b)||_2 <= s`` (row ``QCSTD``, written ``a^2 + b^2 - s^2 <= 0, s >= 0``)
- rotated cone ``a^2 + b^2 <= p * q, p >= 0, q >= 0`` (row ``QCROT``)

.. literalinclude:: examples/socp_sample.mps
   :language: text
   :linenos:

The rotated cone's cross term is given symmetrically as the two half-entries
``P Q -0.5`` and ``Q P -0.5`` (so that ``x^T Q x`` contributes ``-p*q``). Run it
with the same binary:

.. code-block:: bash

   ./mps_file_example socp_sample.mps

cuOpt detects the quadratic constraints, converts them to second-order cones, and
solves with the barrier method. The optimum is ``a = b = 1`` and
``s = p = q = sqrt(2) ≈ 1.414214`` with objective ``3*sqrt(2) ≈ 4.242641``. In the
output below, ``x1..x5`` are ``a, b, s, p, q`` in column order:

.. code-block:: bash
   :caption: Output (barrier iteration log abbreviated)

   Reading and solving MPS file: socp_sample.mps
   Problem has 2 quadratic constraints. Converting to second-order cones and solving with barrier.
   Barrier solver: 5 constraints, 10 variables, 13 nonzeros
   Second-order cones          : 2
   ...
   Optimal solution found in 9 iterations and 0.126s
   Objective +4.24264071e+00
   Barrier finished in 0.13 seconds

   Results:
   --------
   Number of variables: 5
   Termination status: Optimal (1)
   Solve time: 0.128689 seconds
   Objective value: 4.242641

   Primal Solution: First 10 solution variables (or fewer if less exist):
   x1 = 1.000000
   x2 = 1.000000
   x3 = 1.414214
   x4 = 1.414214
   x5 = 1.414214

   Solver completed successfully!
