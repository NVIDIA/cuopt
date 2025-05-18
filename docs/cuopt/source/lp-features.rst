==================
LP Features
==================

Availability
------------------------------

NVIDIA cuOpt LP is available in 3 different ways:

- **Third-Party Modeling Languages**: cuOpt's LP and MILP solver can be called directly from the following third-party modeling languages. This allows you to leverage GPU acceleration while maintaining your existing optimization workflow in these modeling languages.

  Supported modeling languages:
   -  SciPy
   -  PuLP 
   -  CVXPY
   -  Pyomo
   -  AMPL

- **C API**: A native C API that provides direct low-level access to cuOpt's LP capabilities, enabling integration into any application or system that can interface with C.

- **As a Self-Hosted Service**: cuOpt LP can be deployed as a in your own infrastructure, enabling you to maintain full control while integrating it into your existing systems.

All three options provide the same powerful linear programming optimization capabilities while offering flexibility in deployment and integration approaches.

Constraints
-----------

The constraint matrix is specified in `Compressed Sparse Row (CSR) format  <https://docs.nvidia.com/cuda/cusparse/#compressed-sparse-row-csr>`_.

There are two ways to specify constraints in cuOpt LP:

1. Using row_type and right-hand side:

   Constraints can be specified in the form:

   A*x {<=, =, >=} b

   where A is the constraint matrix in CSR format, x is the variable vector, and b is the right-hand side vector. The relationship {<=, =, >=} is specified through the ``row_type`` parameter.

2. Using constraint bounds:

   Alternatively, constraints can be specified as double-sided inequalities:

   lb <= A*x <= ub

   where lb and ub are vectors of lower and upper bounds respectively. This form allows specifying both bounds in a single constraint.

Warm Start
-----------

.. note::
   Warm start is not supported C API and third-party modeling languages.

Warm starts allow a user to provide an initial solution to help PDLP converge faster

For warm start, the initial ``primal`` and ``dual`` solution can be provided to the solver in data.

Alternatively, previously run solutions can be used to warm start a new request to boost the speed to the solution. `Examples <cuopt-server/lp-examples.html#warm-start>`_ are shared on the self-hosted page.

Variable Bounds
---------------

Lower and upper bounds can be applied to each variable. If no variable bounds are specified, the default bounds will be ``[-inf,+inf]``.


PDLP Solver Mode
----------------
Users can control how the solver will operate by using ``solver mode`` under ``solver config``. The mode choice can drastically impact how fast a specific problem will be solved. Users are encouraged to test different modes to see which one fits the best their problem.


Method
------

**Concurrent**: The default method for solving linear programs. When concurrent is selected, cuOpt runs two solves at the same time: PDLP on the GPU and dual simplex on the CPU. A solution is returned from the solve that finishes first.

**PDLP**: Primal-Dual Hybrid Gradient for Linear Program is an algorithm for solving large-scale linear programming problems on the GPU. PDLP does not attempt to any matrix factorizations during the course of the solve. Select this method if your LP is so large that factorization will not fit into memory. By default PDLP solves to low relative tolerance and the solutions it returns do not lie at a vertex of the feasible region. Enable crossover if you need a highly accurate basic solution.

**Dual Simplex**: The simplex method applied to the dual of the linear program. Dual simplex requires the basis factorization of linear program fit into memory. Select this method if your LP is small to medium sized, or if you require a highly accurate basic solution.


Logging Callback
----------------
With logging callback, users can fetch server-side logs for additional debugs and to get details on solver process details. `Examples <cuopt-server/examples/lp-examples.html#logging-callback>`__ are shared on the self-hosted page.


Infeasibility Detection
-----------------------

An option under ``solver config`` in API. The PDLP solver includes the option to detect infeasible problems. If the infeasibilty detection is enabled in solver settings, PDLP will abort as soon as it concludes the problem is infeasible.

Infeasibility detection is always enabled for dual simplex.


Time Limit
----------

The user may specify a time limit to the solver. By default the solver runs until a solution is found or the problem is determined to be infeasible or unbounded.

.. note::

  Note that ``time_limit`` applies only to solve time inside the LP solver. This does not include time for ``network transfer``, ``validation of input``, and other operations that occur outside the solver. The overhead associated with these operations are usually quite small compared to the solve time


Batch Mode
----------

.. note::
   Batch mode is not supported C API and third-party modeling languages. It is only available via cuOpt server.

Users can submit a set of problems which will be solved in a batch. Problems will be solved at the same time in parallel to fully utilize the GPU. Checkout `self-hosted client <cuopt-server/examples/lp-examples.html#batch-mode>`_ example in thin client.

And batch mode is supported only in server.