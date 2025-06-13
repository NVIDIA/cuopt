====================
MILP Features
====================

Availability
------------

The MILP solver can be accessed in the following ways:

- **Third-Party Modeling Languages**: cuOpt's LP and MILP solver can be called directly from the following third-party modeling languages. This allows you to leverage GPU acceleration while maintaining your existing optimization workflow in these modeling languages.

  Currently supported solvers:
   - AMPL
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

Incumbent Solution Callback
---------------------------

User can provide a callback to receive new integer feasible solutions that improve the objective (called incumbents) while the solver is running. An `Incumbent Example <cuopt-server/examples/milp-examples.html#incumbent-solution>`_ is shared on the self-hosted page.

Logging Callback
----------------

A logging callback allows users to get additional information about how the solve is progressing. A `Logging Callback Example <cuopt-server/examples/milp-examples.html#logging-callback>`_ is shared on the self-hosted page.

Time Limit
--------------

The user may specify a time limit to the solver. By default the solver runs until a solution is found or the problem is determined to be infeasible or unbounded.

.. note::

  Note that time_limit applies only to solve time inside the LP solver. This does not include time for network transfer, validation of input, and other operations that occur outside the solver. The overhead associated with these operations are usually small compared to the solve time.
