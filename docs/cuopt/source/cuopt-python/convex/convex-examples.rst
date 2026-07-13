============================
Convex Optimization Examples
============================

This section contains examples of how to use the cuOpt convex optimization Python API.

.. note::

    The examples in this section are not exhaustive. They are provided to help you get started with the cuOpt convex optimization Python API. For more examples, please refer to the `cuopt-examples GitHub repository <https://github.com/NVIDIA/cuopt-examples>`_.


Simple Linear Programming Example
---------------------------------

:download:`simple_lp_example.py <examples/simple_lp_example.py>`

.. literalinclude:: examples/simple_lp_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.01 seconds
    x = 5.0
    y = 5.0
    Objective value = 10.0


.. _simple-qp-example-python:

Simple Quadratic Programming Example
------------------------------------

:download:`simple_qp_example.py <examples/simple_qp_example.py>`

.. literalinclude:: examples/simple_qp_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.01 seconds
    x = 0.5
    y = 0.5
    Objective value = 0.5


.. _simple-socp-example-python:

Second-Order Cone Programming Example
--------------------------------------------

:download:`simple_socp_example.py <examples/simple_socp_example.py>`

This example minimizes ``x3`` subject to ``x1 + x2 >= 2`` and the second-order
cone ``||(x1, x2)||_2 <= x3``, expressed as the inequalities
``x1^2 + x2^2 - x3^2 <= 0, x_3 >= 0``. cuOpt detects the cone structure and solves with the
barrier method.

.. literalinclude:: examples/simple_socp_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    Status: 1
    x1 = 1.0
    x2 = 1.0
    x3 = 1.4142135623730951
    Objective value = 1.4142135623730951


.. _rotated-socp-example-python:

Second-Order Cone Programming with Rotated Second-Order Cones Example
---------------------------------------------------------------------

:download:`rotated_socp_example.py <examples/rotated_socp_example.py>`

This example solves a **rotated** second-order cone ``x1^2 + x2^2 <= x3 * x4, x3 >= 0, x4 >= 0``.
The cross term is written as ``-x3*x4``. See :doc:`/convex-features` for other RSOC forms.
It minimizes ``x3 + x4`` subject to ``x1 + x2 >= 2``.

.. literalinclude:: examples/rotated_socp_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    Status: 1
    x1 = 1.0
    x2 = 1.0
    x3 = 1.4142135623730951
    x4 = 1.4142135623730951
    Objective value = 2.8284271247461903


.. _general-quadratic-example-python:

General Convex Quadratic Constraint Example
-------------------------------------------

:download:`general_quadratic_example.py <examples/general_quadratic_example.py>`

This example uses a general convex quadratic constraint
``2*x^2 + 2*x*y + 2*y^2 <= 6`` (an ellipsoid).  It minimizes ``x + y``.

.. literalinclude:: examples/general_quadratic_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    Status: 1
    x = -1.0
    y = -1.0
    Objective value = -2.0


.. _mps-example-python:

Reading a Problem from an MPS File
----------------------------------

:download:`mps_example.py <examples/mps_example.py>`,
:download:`sample.mps <examples/sample.mps>`

``Problem.read`` loads a problem from an MPS, QPS, or LP file, dispatching on the
file extension. The same call also reads ``QUADOBJ`` quadratic objectives and
and ``QCMATRIX`` quadratic constraints. This example reads the
bundled ``sample.mps`` (a small LP) and solves it.

.. literalinclude:: examples/mps_example.py
   :language: python
   :linenos:

The sample MPS file:

.. literalinclude:: examples/sample.mps
   :language: text
   :linenos:

The response is as follows:

.. code-block:: text

    Status: 1
    Number of variables: 2
    Objective value = -0.36000000000000004
    VAR1 = 1.8
    VAR2 = 0.0


Advanced Example: Production Planning
-------------------------------------

:download:`production_planning_example.py <examples/production_planning_example.py>`

.. literalinclude:: examples/production_planning_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    === Production Planning Solution ===

    Status: Optimal
    Solve time: 0.09 seconds
    Product A production: 36.0 units
    Product B production: 28.000000000000004 units
    Total profit: $2640.00

Working with Expressions and Constraints
----------------------------------------

:download:`expressions_constraints_example.py <examples/expressions_constraints_example.py>`

.. literalinclude:: examples/expressions_constraints_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    === Expression Example Results ===
    x = 0.0
    y = 50.0
    z = 100.0
    Objective value = 400.0

Working with Quadratic Objective Matrix
---------------------------------------

:download:`qp_matrix_example.py <examples/qp_matrix_example.py>`

.. literalinclude:: examples/qp_matrix_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

   Optimal solution found in 0.16 seconds
   p1 = 30.770728122083014
   p2 = 65.38350784293876
   p3 = 53.84576403497824
   Minimized cost = 1153.8461538953868

Inspecting the Problem Solution
-------------------------------

:download:`solution_example.py <examples/solution_example.py>`

.. literalinclude:: examples/solution_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.02 seconds
    Objective: 9.0
    x = 1.0, ReducedCost = 0.0
    y = 3.0, ReducedCost = 0.0
    z = 0.0, ReducedCost = 2.999999858578644
    c1 DualValue = 1.0000000592359144
    c2 DualValue = 1.0000000821854418

Working with PDLP Warmstart Data
--------------------------------

Warmstart data allows to restart PDLP with a previous solution context. This should be used when you solve a new problem which is similar to the previous one.

.. note::
    Warmstart data is only available for Linear Programming (LP) problems, not for Mixed Integer Linear Programming (MILP) problems.

:download:`pdlp_warmstart_example.py <examples/pdlp_warmstart_example.py>`

.. literalinclude:: examples/pdlp_warmstart_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.01 seconds
    x = 25.001763214569394
    y = 0.0
    Objective value = 50.00352642913879
