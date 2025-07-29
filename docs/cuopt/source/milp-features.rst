====================
MILP Features
====================

Availability
------------

The MILP solver can be accessed in the following ways:

- **Third-Party Modeling Languages**: cuOpt's LP and MILP solver can be called directly from the following third-party modeling languages. This allows you to leverage GPU acceleration while maintaining your existing optimization workflow in these modeling languages.

  Currently supported solvers:
   - AMPL
   - GAMS
   - PuLP

- **C API**: A native C API that provides direct low-level access to cuOpt's MILP solver, enabling integration into any application or system that can interface with C.

- **As a Self-Hosted Service**: cuOpt's MILP solver can be deployed in your own infrastructure, enabling you to maintain full control while integrating it into your existing systems.

Each option provide the same powerful mixed-integer linear optimization capabilities while offering flexibility in deployment and integration.

Variable Bounds
---------------

Lower and upper bounds can be applied to each variable. If no variable bounds are specified, the default bounds are ``[-inf,+inf]``.

Constraints
-----------

The constraint matrix is specified in `Compressed Sparse Row (CSR) format  <https://docs.nvidia.com/cuda/cusparse/#compressed-sparse-row-csr>`_.

There are two ways to specify constraints in cuOpt MILP:

1. Using row_type and right-hand side:

   Constraints can be specified in the form:

   A*x {<=, =, >=} b

   where A is the constraint matrix in CSR format, x is the variable vector, and b is the right-hand side vector. The relationship {<=, =, >=} is specified through the ``row_type`` parameter.

2. Using constraint bounds:

   Alternatively, constraints can be specified as two-sided inequalities:

   lb <= A*x <= ub

   where lb and ub are vectors of lower and upper bounds respectively. This form allows specifying both bounds of a single constraint.

Both forms are mathematically equivalent. The choice between them is a matter of convenience depending on your problem formulation.

Incumbent Solution Callback in the Service
------------------------------------------

When using the service, users can provide a callback to receive new integer feasible solutions that improve the objective (called incumbents) while the solver is running. An :ref:`Incumbent Example <incumbent-and-logging-callback>` is shared on the self-hosted page.

Logging
-------

The CUOPT_LOG_FILE parameter can be set to write detailed solver logs for MILP problems. This parameter is available in all APIs that allow setting solver parameters except for the cuOpt service. For the service, see the logging callback below.

Logging Callback in the Service
-------------------------------

In the cuOpt service API, the ``log_file`` value in ``solver_configs`` is ignored.

If however you set the ``solver_logs`` flag on the ``/cuopt/request`` REST API call, users can fetch the log file content from the webserver at ``/cuopt/logs/{id}``. Using the logging callback feature through the cuOpt client is shown in :ref:`Logging Callback Example <incumbent-and-logging-callback>` on the self-hosted page.


Time Limit
--------------

The user may specify a time limit to the solver. By default the solver runs until a solution is found or the problem is determined to be infeasible or unbounded.

.. note::

  Note that time_limit applies only to solve time inside the LP solver. This does not include time for network transfer, validation of input, and other operations that occur outside the solver. The overhead associated with these operations are usually small compared to the solve time.
