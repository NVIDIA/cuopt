========================
LP, QP and SOCP Examples
========================

This section contains examples of how to use the cuOpt LP, QP, and SOCP Python API.

.. note::

    The examples in this section are not exhaustive. They are provided to help you get started with the cuOpt LP, QP, and SOCP Python API. For more examples, please refer to the `cuopt-examples GitHub repository <https://github.com/NVIDIA/cuopt-examples>`_.


Simple Linear Programming Example
---------------------------------

:download:`simple_lp_example.py <examples/simple_lp_example.py>`

.. literalinclude:: examples/simple_lp_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.01 seconds
    x = 10.0
    y = 0.0
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

Simple Second-Order Cone Programming Example
--------------------------------------------

:download:`simple_socp_example.py <examples/simple_socp_example.py>`

This example minimizes ``x3`` subject to ``x1 + x2 >= 2`` and the second-order
cone ``||(x1, x2)||_2 <= x3``, expressed as the quadratic inequality
``x1^2 + x2^2 - x3^2 <= 0``. cuOpt detects the cone structure and solves with the
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
    z = 99.99999999999999
    Objective value = 399.99999999999994

Working with Quadratic objective matrix
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
    x = 25.000000000639382
    y = 0.0
    Objective value = 50.000000001278764
