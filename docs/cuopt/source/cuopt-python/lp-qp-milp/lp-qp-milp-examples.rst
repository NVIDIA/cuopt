========================
LP, QP and MILP Examples
========================

This section contains examples of how to use the cuOpt linear programming, quadratic programming and mixed integer linear programming Python API.

.. note::

    The examples in this section are not exhaustive. They are provided to help you get started with the cuOpt linear programming, quadratic programming and mixed integer linear programming Python API. For more examples, please refer to the `cuopt-examples GitHub repository <https://github.com/NVIDIA/cuopt-examples>`_.


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

.. note::
   The QP solver is currently in beta.

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


Mixed Integer Linear Programming Example
----------------------------------------

:download:`simple_milp_example.py <examples/simple_milp_example.py>`

.. literalinclude:: examples/simple_milp_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.00 seconds
    x = 36.0
    y = 40.99999999999999
    Objective value = 303.0


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

Working with Incumbent Solutions
--------------------------------

Incumbent solutions are intermediate feasible solutions found during the MIP solving process. They represent the best integer-feasible solution discovered so far and can be accessed through callback functions.

.. note::
    Incumbent solutions are only available for Mixed Integer Programming (MIP) problems, not for pure Linear Programming (LP) problems.

:download:`incumbent_solutions_example.py <examples/incumbent_solutions_example.py>`

.. literalinclude:: examples/incumbent_solutions_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    Optimal solution found.
    Incumbent 1: [ 0. 58.], cost: 174.00
    Incumbent 2: [36. 41.], cost: 303.00
    Generated fast solution in 0.158467 seconds with objective 303.000000
    Consuming B&B solutions, solution queue size 2
    Solution objective: 303.000000 , relative_mip_gap 0.000000 solution_bound 303.000000 presolve_time 0.043211 total_solve_time 0.160270 max constraint violation 0.000000 max int violation 0.000000 max var bounds violation 0.000000 nodes 4 simplex_iterations 3

    === Final Results ===
    Problem status: Optimal
    Solve time: 0.16 seconds
    Final solution: x=36.0, y=40.99999999999999
    Final objective value: 303.00

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
