==================
LP Features
==================

Availability
-------------

The LP solver can be accessed in the following ways:

- **Third-Party Modeling Languages**: cuOpt's LP and MILP solver can be called directly from the following third-party modeling languages. This allows you to leverage GPU acceleration while maintaining your existing optimization workflow in these modeling languages.

  Supported modeling languages:
   -  AMPL
   -  PuLP 

- **C API**: A native C API that provides direct low-level access to cuOpt's LP capabilities, enabling integration into any application or system that can interface with C.

- **As a Self-Hosted Service**: cuOpt's LP solver can be deployed as a in your own infrastructure, enabling you to maintain full control while integrating it into your existing systems.

Each option provide the same powerful linear optimization capabilities while offering flexibility in deployment and integration.

Variable Bounds
---------------

Lower and upper bounds can be applied to each variable. If no variable bounds are specified, the default bounds are ``[-inf,+inf]``.

Constraints
-----------

The constraint matrix is specified in `Compressed Sparse Row (CSR) format  <https://docs.nvidia.com/cuda/cusparse/#compressed-sparse-row-csr>`_.

There are two ways to specify constraints to the LP solver:

1. Using row_type and right-hand side:

   Constraints can be specified in the form:

   A*x {<=, =, >=} b

   where A is the constraint matrix in CSR format, x is the variable vector, and b is the right-hand side vector. The relationship {<=, =, >=} is specified via the ``row_type`` parameter.

2. Using constraint bounds:

   Alternatively, constraints can be specified as two-sided inequalities:

   lb <= A*x <= ub

   where lb and ub are vectors of lower and upper bounds respectively. This form allows specifying both bounds on a single constraint.

Warm Start
-----------

A warm starts allow a user to provide an initial solution to help PDLP converge faster. The initial ``primal`` and ``dual`` solutions can be specified by the user.

Alternatively, previously run solutions can be used to warm start a new solve to decrease solve time. `Examples <cuopt-server/lp-examples.html#warm-start>`_ are shared on the self-hosted page.

PDLP Solver Mode
----------------
Users can control how the solver will operate by specifying the PDLP solver mode. The mode choice can drastically impact how fast a specific problem will be solved. Users are encouraged to test different modes to see which one fits the best their problem.


Method
------

**Concurrent**: The default method for solving linear programs. When concurrent is selected, cuOpt runs two algorithms at the same time: PDLP on the GPU and dual simplex on the CPU. A solution is returned from the algorithm that finishes first.

**PDLP**: Primal-Dual Hybrid Gradient for Linear Program is an algorithm for solving large-scale linear programming problems on the GPU. PDLP does not attempt to any matrix factorizations during the course of the solve. Select this method if your LP is so large that factorization will not fit into memory. By default PDLP solves to low relative tolerance and the solutions it returns do not lie at a vertex of the feasible region. Enable crossover to obtain a highly accurate basic solution from a PDLP solution.

**Dual Simplex**: Dual simplex is the simplex method applied to the dual of the linear program. Dual simplex requires the basis factorization of linear program fit into memory. Select this method if your LP is small to medium sized, or if you require a high-quality basic solution.


Crossover
---------

Crossover allows you to obtain a high-quality basic solution from the results of a PDLP solve. More details can be found `here <lp-milp-settings.html#crossover>`__.


Logging Callback
----------------
With logging callback, users can fetch server-side logs for additional debugs and to get details on solver process details. `Examples <cuopt-server/examples/lp-examples.html#logging-callback>`__ are shared on the self-hosted page.


Infeasibility Detection
-----------------------

The PDLP solver includes the option to detect infeasible problems. If the infeasibilty detection is enabled in solver settings, PDLP will abort as soon as it concludes the problem is infeasible.

.. note::
   Infeasibility detection is always enabled for dual simplex.

Time Limit
----------

The user may specify a time limit to the solver. By default the solver runs until a solution is found or the problem is determined to be infeasible or unbounded.

.. note::

  Note that ``time_limit`` applies only to solve time inside the LP solver. This does not include time for network transfer, validation of input, and other operations that occur outside the solver. The overhead associated with these operations are usually small compared to the solve time.


Batch Mode
----------

Users can submit a set of problems which will be solved in a batch. Problems will be solved at the same time in parallel to fully utilize the GPU. Checkout `self-hosted client <cuopt-server/examples/lp-examples.html#batch-mode>`_ example in thin client.
