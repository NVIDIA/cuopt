==================
LP Features
==================

Availability
-------------

The LP solver can be accessed in the following ways:

- **Third-Party Modeling Languages**: cuOpt's LP and MILP solver can be called directly from the following third-party modeling languages. This allows you to leverage GPU acceleration while maintaining your existing optimization workflow in these modeling languages.

  Supported modeling languages:
   -  AMPL
   -  GAMS
   -  PuLP
   -  JuMP

- **C API**: A native C API that provides direct low-level access to cuOpt's LP capabilities, enabling integration into any application or system that can interface with C.

- **Python SDK**: A Python package that provides direct access to cuOpt's LP capabilities through a simple, intuitive API. This allows for seamless integration into Python applications and workflows. For more information, see :doc:`cuopt-python/quick-start`.

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

Alternatively, previously run solutions can be used to warm start a new solve to decrease solve time. :ref:`Examples <warm-start>` are shared on the self-hosted page.

PDLP Solver Mode
----------------
Users can control how the solver will operate by specifying the PDLP solver mode. The mode choice can drastically impact how fast a specific problem will be solved. Users are encouraged to test different modes to see which one fits the best their problem.


Method
------

**Concurrent**: The default method for solving linear programs. When concurrent is selected, cuOpt runs three algorithms in parallel: PDLP on the GPU, barrier (interior-point) on the GPU, and dual simplex on the CPU. A solution is returned from the algorithm that finishes first.

**PDLP**: Primal-Dual Hybrid Gradient for Linear Program is an algorithm for solving large-scale linear programming problems on the GPU. PDLP does not attempt any matrix factorizations during the course of the solve. Select this method if your LP is so large that factorization will not fit into memory. By default PDLP solves to low relative tolerance and the solutions it returns do not lie at a vertex of the feasible region. Enable crossover to obtain a highly accurate basic solution from a PDLP solution.

**Barrier**: The barrier method (also known as interior-point method) solves linear programs using a primal-dual predictor-corrector algorithm. This method uses GPU-accelerated sparse Cholesky and sparse LDLT solves via cuDSS, and GPU-accelerated sparse matrix-vector and matrix-matrix operations via cuSparse. Barrier is particularly effective for large-scale problems and can automatically apply techniques like folding, dualization, and dense column elimination to improve performance. This method solves the linear systems at each iteration using the augmented system or the normal equations (ADAT). Enable crossover to obtain a highly accurate basic solution from a barrier solution.

**Dual Simplex**: Dual simplex is the simplex method applied to the dual of the linear program. Dual simplex requires the basis factorization of linear program fit into memory. Select this method if your LP is small to medium sized, or if you require a high-quality basic solution.


Crossover
---------

Crossover allows you to obtain a high-quality basic solution from the results of a PDLP or barrier solve. When enabled, crossover converts these solutions to a vertex solution (basic solution) with high accuracy. More details can be found :ref:`here <crossover>`.


Presolve
--------

Presolve procedure is applied to the problem before the solver is called. It can be used to reduce the problem size and improve solve time. It is enabled by default for MIP problems, and disabled by default for LP problems.
Furthermore, for LP problems, when the dual solution is not needed, additional presolve procedures can be applied to further improve solve times. This is achived by turned off dual postsolve.


Logging
-------

The CUOPT_LOG_FILE parameter can be set to write detailed solver logs for LP problems. This parameter is available in all APIs that allow setting solver parameters except the cuOpt service. For the service, see the logging callback below.

Logging Callback in the Service
-------------------------------

In the cuOpt service API, the ``log_file`` value in ``solver_configs`` is ignored.

If however you set the ``solver_logs`` flag on the ``/cuopt/request`` REST API call, users can fetch the log file content from the webserver at ``/cuopt/logs/{id}``. Using the logging callback feature through the cuOpt client is shown in :ref:`Examples <generic-example-with-normal-and-batch-mode>` on the self-hosted page.


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

Users can submit a set of problems which will be solved in a batch. Problems will be solved at the same time in parallel to fully utilize the GPU. Checkout :ref:`self-hosted client <generic-example-with-normal-and-batch-mode>` example in thin client.
