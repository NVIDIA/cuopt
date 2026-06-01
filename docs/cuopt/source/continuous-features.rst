========================
LP/QP/SOCP Features
========================

Availability
-------------

The Linear Programming (LP), Quadratic Programming (QP), and Second-Order Cone Programming (SOCP) solvers can be accessed in the following ways:

- **Third-Party Modeling Languages**: cuOpt's LP/QP solver can be called directly from the following third-party modeling languages. This allows you to leverage GPU acceleration while maintaining your existing optimization workflow in these modeling languages.

  Supported modeling languages:

  .. list-table::
     :header-rows: 1
     :widths: 30 15 15

     * - Language
       - LP
       - QP
     * - AMPL
       - ✓
       -
     * - CVXPY
       - ✓
       - ✓
     * - GAMS
       - ✓
       - ✓
     * - JuMP
       - ✓
       - ✓
     * - PuLP
       - ✓
       -

.. note::
   SOCP is not yet supported in third-party modeling languages.

- **C API**: A native C API that provides direct low-level access to cuOpt's LP/QP/SOCP capabilities, enabling integration into any application or system that can interface with C.

- **Python SDK**: A Python package that provides direct access to cuOpt's LP/QP/SOCP capabilities through a simple, intuitive API. This allows for seamless integration into Python applications and workflows. For more information, see :doc:`cuopt-python/quick-start`.

- **As a Self-Hosted Service**: cuOpt's LP/QP solver can be deployed as a self-hosted service in your own infrastructure, enabling you to maintain full control while integrating it into your existing systems.

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


Quadratic Programming
---------------------

cuOpt supports problems with quadratic objectives of the form:

.. code-block:: text

    minimize        x^T*Q*x + c^T*x
    subject to      A*x {<=, =, >=} b
                    lb <= x <= ub

where Q is a symmetric positive semidefinite matrix. Please note that the Q matrix is specified without the 1/2 factor that may be used by other solvers.

.. note:: Currently, barrier is the only method that supports QPs.

See :ref:`simple-qp-example-python` for an example of how to create a QP problem with the Python Modeling API.
See :ref:`simple-qp-example-c` for an example of how to create a QP problem with the C API.

Second-Order Cone Programming (Beta)
--------------------------------------

.. note:: SOCP support is **beta** in this release.

cuOpt accepts **quadratic constraints** of the form

.. code-block:: text

    x^T Q x + c^T x + alpha <= 0

and translates them internally into **second-order cone** constraints, so it can solve second-order cone programs (SOCP). A problem then has the shape:

.. code-block:: text

    minimize        c^T*x
    subject to      A*x {<=, =, >=} b
                    x^T Q_i x + c_i^T x + alpha_i <= 0   (quadratic constraints)
                    lb <= x <= ub

Quadratic constraints are supplied through the quadratic constraint API — in Python via ``addConstraint`` with a quadratic expression, in C via :c:func:`cuOptAddQuadraticConstraint`, and in MPS via ``QCMATRIX`` sections — rather than as cone constraints directly. cuOpt detects the second-order cone structure of the quadratic form and converts it before solving with the barrier method.

.. note::
   Only quadratic constraints whose ``(Q, c, alpha)`` describe a second-order cone are
   currently supported. General quadratic constraints (arbitrary quadratic forms) are
   not supported.

In practice the supported quadratic forms are the two cone families below, written with a zero right-hand side. ``x_1, ..., x_k`` denote the variables that participate in a single constraint:

**Standard (Lorentz) cone** — ``||(x_1, ..., x_{k-1})||_2 <= x_k`` with ``x_k >= 0``. Express it as the quadratic inequality

.. code-block:: text

    x_1^2 + ... + x_{k-1}^2 - x_k^2 <= 0

**Rotated cone** — ``x_1^2 + ... + x_{k-2}^2 <= x_{k-1} * x_k`` with ``x_{k-1} >= 0`` and ``x_k >= 0``. Express it as the quadratic inequality

.. code-block:: text

    x_1^2 + ... + x_{k-2}^2 - x_{k-1} * x_k <= 0

When any quadratic constraint is present, cuOpt automatically selects the barrier method and disables presolve optimizations that apply only to linear problems.

**Constraints:**

- Only ``<=`` and ``>=`` sense is supported. Equality quadratic constraints are not supported.
- The right-hand side must be ``0``.
- The quadratic form must have valid second-order cone structure (standard or rotated).

**Python example — standard cone** ``||(x_1, x_2)||_2 <= x_3``:

.. code-block:: python

    x1 = problem.addVariable(name="x1")
    x2 = problem.addVariable(name="x2")
    x3 = problem.addVariable(name="x3", lb=0)   # cone head, must be >= 0

    # ||(x1, x2)||_2 <= x3  expressed as  x1^2 + x2^2 - x3^2 <= 0
    problem.addConstraint(x1*x1 + x2*x2 - x3*x3 <= 0, name="soc")

**Python example — rotated cone** ``x_1^2 + x_2^2 <= x_3 * x_4``:

.. code-block:: python

    x1 = problem.addVariable(name="x1")
    x2 = problem.addVariable(name="x2")
    x3 = problem.addVariable(name="x3", lb=0)   # cone heads, must be >= 0
    x4 = problem.addVariable(name="x4", lb=0)

    # x1^2 + x2^2 <= x3 * x4  expressed as  x1^2 + x2^2 - x3*x4 <= 0.
    # The quadratic form must be symmetric, so the cross term is supplied as the
    # two equal halves -0.5*x3*x4 and -0.5*x4*x3 (i.e. Q[x3,x4] = Q[x4,x3] = -0.5).
    problem.addConstraint(
        x1*x1 + x2*x2 - 0.5*x3*x4 - 0.5*x4*x3 <= 0, name="rotated_soc"
    )

.. note::
   The rotated-cone cross term must be written symmetrically (both ``x3*x4`` and
   ``x4*x3`` halves). Supplying only ``- x3*x4`` produces an asymmetric quadratic
   form that is not recognized as a second-order cone.

**C API:** Use :c:func:`cuOptAddQuadraticConstraint` to add standard or rotated SOC constraints expressed as quadratic inequalities. cuOpt detects the SOC structure and converts to cone form internally.

.. note:: SOCP problems always use the barrier solver regardless of the ``CUOPT_METHOD`` setting.

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

.. note::
   PDLP solves to 1e-4 relative accuracy by default.

**Barrier**: The barrier method (also known as interior-point method) solves linear and quadratic programs using a primal-dual predictor-corrector algorithm. This method uses GPU-accelerated sparse Cholesky and sparse LDLT solves via cuDSS, and GPU-accelerated sparse matrix-vector and matrix-matrix operations via cuSparse. Barrier is particularly effective for large-scale problems and can automatically apply techniques like folding, dualization, and dense column elimination to improve performance. This method solves the linear systems at each iteration using the augmented system or the normal equations (ADAT). Enable crossover to obtain a highly accurate basic solution from a barrier solution.

.. note::
   Barrier solves to 1e-8 relative accuracy by default.

**Dual Simplex**: Dual simplex is the simplex method applied to the dual of the linear program. Dual simplex requires the basis factorization of linear program fit into memory. Select this method if your LP is small to medium sized, or if you require a high-quality basic solution.

.. note::
   Dual Simplex solves to 1e-6 absolute accuracy by default.


Crossover
---------

Crossover allows you to obtain a high-quality basic solution from the results of a PDLP or barrier solve. When enabled, crossover converts the PDLP or barrier solution to a vertex solution (basic solution) with high accuracy. More details can be found :ref:`here <crossover>`.

.. note::
   Crossover is not supported for QP problems.

Presolve
--------

Presolve procedure is applied to the problem before the solver is called. It can be used to reduce the problem size and improve solve time. cuOpt supports presolve reductions using PSLP or Papilo for linear programming (LP) problems. For LP problems, PSLP presolve is always enabled by default. Users can manually select to disable presolve by setting this parameter to 0, enable Papilo presolve by setting this parameter to 1, or enable PSLP presolve by setting this parameter to 2.
Furthermore, for LP problems with Papilo presolver, when the dual solution is not needed, additional presolve procedures can be applied to further improve solve times. This is achieved by turning off dual postsolve with the ``CUOPT_DUAL_POSTSOLVE`` setting.


Logging
-------

The ``CUOPT_LOG_FILE`` parameter can be set to write detailed solver logs for LP/QP problems. This parameter is available in all APIs that allow setting solver parameters except the cuOpt service. For the service, see the logging callback below.

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


.. _batch-mode:

Batch Mode
----------

Users can submit a set of problems which will be solved in a batch. Problems will be solved at the same time in parallel to fully utilize the GPU. Checkout :ref:`self-hosted client <generic-example-with-normal-and-batch-mode>` example in thin client.

.. warning:: Deprecated

   LP batch mode (Python ``cuopt.linear_programming.BatchSolve``, server requests
   with a list of LP problems, and multi-file ``cuopt_sh`` LP submissions) is
   deprecated and will be removed in a future release. Prefer sequential
   ``cuopt.linear_programming.Solve`` calls, or implement your own parallelism
   (for example with ``concurrent.futures``). Existing batch APIs still run in
   parallel today; callers may see a ``DeprecationWarning`` or a deprecation
   message in server ``warnings``.

PDLP Precision Modes
--------------------

By default, PDLP operates in the native precision of the problem type (FP64 for double-precision problems). The ``pdlp_precision`` parameter provides several modes:

- **single**: Run PDLP internally in FP32, with automatic conversion of inputs and outputs. FP32 uses half the memory and allows PDHG iterations to be on average twice as fast, but may require more iterations to converge. Compatible with crossover (solution is converted back to FP64 before crossover) and concurrent mode (PDLP runs in FP32 while other solvers run in FP64).
- **mixed**: Use mixed precision SpMV during PDHG iterations. The constraint matrix is stored in FP32 while vectors and compute type remain in FP64, improving SpMV performance with limited impact on convergence. Convergence checking and restart logic always use the full FP64 matrix.
- **double**: Explicitly run in FP64 (same as default for double-precision problems).

.. note:: The default precision is the native type of the problem (FP64 for double).

Multi-GPU Mode
--------------

Users can use multiple GPUs to solve a problem by specifying the ``num_gpus`` parameter. The feature is restricted to LP problems that uses concurrent mode and supports up to 2 GPUs at the moment. Using this mode will run PDLP and barrier in parallel on different GPUs to avoid sharing single GPU resources.
