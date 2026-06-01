============
MIP Examples
============

This section contains examples of how to use the cuOpt MIP Python API.

.. note::

    The examples in this section are not exhaustive. They are provided to help you get started with the cuOpt mixed integer linear programming Python API. For more examples, please refer to the `cuopt-examples GitHub repository <https://github.com/NVIDIA/cuopt-examples>`_.


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
    y = 41.0
    Objective value = 303.0


Semi-continuous Variable Example
--------------------------------

:download:`semi_continuous_example.py <examples/semi_continuous_example.py>`

.. literalinclude:: examples/semi_continuous_example.py
   :language: python
   :linenos:

The response is as follows:

.. code-block:: text

    Optimal solution found in 0.00 seconds
    x = 0.0
    y = 1.0
    Objective value = 0.0


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
    Incumbent 1: x=36.0 y=41.0 cost: 303.00
    Solution objective: 303.000000 , relative_mip_gap 0.000000 solution_bound 303.000000 presolve_time 0.103659 total_solve_time 0.173678 max constraint violation 0.000000 max int violation 0.000000 max var bounds violation 0.000000 nodes 0 simplex_iterations 2

    === Final Results ===
    Problem status: Optimal
    Solve time: 0.17 seconds
    Final solution:  x=36.0  y=41.0
    Final objective value: 303.00

    Total incumbent solutions found: 1
