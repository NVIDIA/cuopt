=================================
LP and MILP Settings
=================================


This page describes the parameter settings available for cuOpt's LP and MILP solvers. These parameters are set as :ref:`parameter constants <parameter-constants>` in case of C API and in case of Server Thin client as raw strings.
Please refer to examples in :doc:`C </cuopt-c/lp-milp/index>` and :doc:`Server Thin client </cuopt-server/index>` for more details.

.. note::
   When setting parameters in thin client solver settings, remove ``CUOPT_`` from the parameter name and convert to lowercase. For example, ``CUOPT_TIME_LIMIT`` would be set as ``time_limit``.

Parameters common to LP/MILP
----------------------------

We begin by describing parameters common to both the MILP and LP solvers


Time Limit
^^^^^^^^^^
``CUOPT_TIME_LIMIT`` controls the time limit in seconds after which the solver will stop and return the current solution.
For performance reasons, cuOpt does not constantly checks for time limit. Thus, the solver
may run slightly over the limit. If set along with the iteration limit, cuOpt will stop when
the first limit (iteration or time) is hit.


.. note:: by default there is no time limit. So cuOpt will run until it finds an optimal solution,
   or proves the problem is infeasible or unbounded.



Log to Console
^^^^^^^^^^^^^^
``CUOPT_LOG_TO_CONSOLE`` controls whether cuOpt should log information to the console during a solve.
If true, a logging info is written to the console, if false no logging info is written to the console (logs may still be written to a file.)

.. note:: the default value is true.

Log File
^^^^^^^^
``CUOPT_LOG_FILE`` controls the name of a log file where cuOpt should write information about the solve.

.. note:: the default value is ``""`` and no log file is written. This setting is ignored by the cuOpt service, use the log callback feature instead.

Solution File
^^^^^^^^^^^^^
``CUOPT_SOLUTION_FILE`` controls the name of a file where cuOpt should write the solution.

.. note:: the default value is ``""`` and no solution file is written. This setting is ignored by the cuOpt service.

User Problem File
^^^^^^^^^^^^^^^^^
``CUOPT_USER_PROBLEM_FILE`` controls the name of a file where cuOpt should write the user problem.

.. note:: the default value is ``""`` and no user problem file is written. This setting is ignored by the cuOpt service.

Num CPU Threads
^^^^^^^^^^^^^^^
``CUOPT_NUM_CPU_THREADS`` controls the number of CPU threads used in the LP and MIP solvers. Set this to a small value to limit
the amount of CPU resources cuOpt uses. Set this to a large value to improve solve times for CPU
parallel parts of the solvers.

.. note:: by default the number of CPU threads is automatically determined based on the number of CPU cores.

Presolve
^^^^^^^^
``CUOPT_PRESOLVE`` controls whether presolve is enabled. Presolve can reduce problem size and improve solve time. Enabled by default for MIP, disabled by default for LP.

Dual Postsolve
^^^^^^^^^^^^^^
``CUOPT_DUAL_POSTSOLVE`` controls whether dual postsolve is enabled. Disabling dual postsolve can improve solve time at the expense of not having
access to the dual solution. Enabled by default for LP when presolve is enabled. This is not relevant for MIP problems.

Linear Programming
------------------

We now describe the parameter settings used to control cuOpt's Linear Programming solvers

Method
^^^^^^

``CUOPT_METHOD`` controls the method to solve the linear programming problem. Four methods are available:

* ``Concurrent``: Use PDLP, dual simplex, and barrier in parallel (default).
* ``PDLP``: Use the PDLP method.
* ``Dual Simplex``: Use the dual simplex method.
* ``Barrier``: Use the barrier (interior-point) method.

.. note:: The default method is ``Concurrent``.

Default accuracy for each method:

* PDLP solves to 1e-4 relative accuracy by default.
* Barrier solves to 1e-8 relative accuracy by default.
* Dual Simplex solves to 1e-6 absolute accuracy by default.

C API users should use the constants defined in :ref:`method-constants` for this parameter.

Server Thin client users should use the :class:`cuopt_sh_client.SolverMethod` for this parameter.

PDLP Solver Mode
^^^^^^^^^^^^^^^^

``CUOPT_PDLP_MODE`` controls the mode under which PDLP should operate. The mode will change the way the
PDLP internally optimizes the problem. The mode choice can drastically impact how fast a
specific problem will be solved. Users are encouraged to test different modes to see which one
fits the best their problem. By default, the solver uses ``Stable3``, the best
overall mode from our experiments. For now, only three modes are available: ``Stable3``,
``Methodical1``, and ``Fast1``.

For now, we do not offer a mechanism to know upfront which solver mode will be the best
for a specific problem.

C API users should use the constants defined in :ref:`pdlp-solver-mode-constants` for this parameter.

Server Thin client users should use the :class:`cuopt_sh_client.PDLPSolverMode` for this parameter.

Iteration Limit
^^^^^^^^^^^^^^^

``CUOPT_ITERATION_LIMIT`` controls the iteration limit after which the solver will stop and return the current solution.
For performance reasons, cuOpt's does not constantly checks for iteration limit, thus,
the solver might run a few extra iterations over the limit. If set along with the time limit,
cuOpt will stop at the first limit (iteration or time) reached.

.. note:: by default there is no iteration limit. So, cuOpt will run until it finds an optimal solution,
   or proves the problem is infeasible or unbounded.


Infeasiblity Detection
^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_INFEASIBILITY_DETECTION`` controls whether PDLP should detect infeasibility. Note that infeasibility detection in PDLP
is not always accurate. Some problems detected as infeasible may converge under a different tolerance factor.
Detecting infeasibility consumes both more runtime and memory. The added runtime is between 3% and 7%,
the added memory consumpution is between 10% and 20%.

.. note:: by default PDLP will not detect infeasibility. Dual simplex will always detect infeasibility
   regardless of this setting.

Strict Infeasibility
^^^^^^^^^^^^^^^^^^^^

``CUOPT_STRICT_INFEASIBILITY`` controls the strict infeasibility mode in PDLP. When true if either the current or the average solution
is detected as infeasible, PDLP will stop. When false both the current and average solution need to be
detected as infeasible for PDLP to stop.

.. note:: the default value is false.

.. _crossover:

Crossover
^^^^^^^^^

``CUOPT_CROSSOVER`` controls whether PDLP or barrier should crossover to a basic solution after an optimal solution is found.
Changing this value has a significant impact on accuracy and runtime.
By default the solutions provided by PDLP and barrier do not lie at a vertex and thus may have many variables that lie
between their bounds. Enabling crossover allows the user to obtain a high-quality basic solution
that lies at a vertex of the feasible region. If n is the number of variables, and m is the number of
constraints, n - m variables will be on their bounds in a basic solution.

.. note:: the default value is false.

Save Best Primal So Far
^^^^^^^^^^^^^^^^^^^^^^^
``CUOPT_SAVE_BEST_PRIMAL_SOLUTION`` controls whether PDLP should save the best primal solution so far
With this parameter set to true, PDLP
* Will always prioritize a primal feasible to a non primal feasible
* If a new primal feasible is found, the one with the best primal objective will be kept
* If no primal feasible was found, the one with the lowest primal residual will be kept
* If two have the same primal residual, the one with the best objective will be kept

.. note:: the default value is false.

First Primal Feasible
^^^^^^^^^^^^^^^^^^^^^

``CUOPT_FIRST_PRIMAL_FEASIBLE`` controls whether PDLP should stop when the first primal feasible solution is found.

.. note:: the default value is false.

Per Constraint Residual
^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_PER_CONSTRAINT_RESIDUAL`` controls whether PDLP should compute the primal & dual residual per constraint instead of globally.

.. note:: the default value is false.

Barrier Solver Settings
^^^^^^^^^^^^^^^^^^^^^^^^

The following settings control the behavior of the barrier (interior-point) method:

Folding
"""""""

``CUOPT_FOLDING`` controls whether to fold the linear program. Folding can reduce problem size by exploiting symmetry in the problem.

* ``-1``: Automatic (default) - cuOpt decides whether to fold based on problem characteristics
* ``0``: Disable folding
* ``1``: Force folding to run

.. note:: the default value is ``-1`` (automatic).

Dualize
"""""""

``CUOPT_DUALIZE`` controls whether to dualize the linear program in presolve. Dualizing can improve solve time for problems, with inequality constraints, where there are more constraints than variables.

* ``-1``: Automatic (default) - cuOpt decides whether to dualize based on problem characteristics
* ``0``: Don't attempt to dualize
* ``1``: Force dualize

.. note:: the default value is ``-1`` (automatic).

Ordering
""""""""

``CUOPT_ORDERING`` controls the ordering algorithm used by cuDSS for sparse factorizations. The ordering can significantly impact solver run time.

* ``-1``: Automatic (default) - cuOpt selects the best ordering
* ``0``: cuDSS default ordering
* ``1``: AMD (Approximate Minimum Degree) ordering

.. note:: the default value is ``-1`` (automatic).

Augmented System
""""""""""""""""

``CUOPT_AUGMENTED`` controls which linear system to solve in the barrier method.

* ``-1``: Automatic (default) - cuOpt selects the best linear system to solve
* ``0``: Solve the ADAT system (normal equations)
* ``1``: Solve the augmented system

.. note:: the default value is ``-1`` (automatic). The augmented system may be more stable for some problems, while ADAT may be faster for others.

Eliminate Dense Columns
""""""""""""""""""""""""

``CUOPT_ELIMINATE_DENSE_COLUMNS`` controls whether to eliminate dense columns from the constraint matrix before solving. Eliminating dense columns can improve performance by reducing fill-in during factorization.
However, extra solves must be performed at each iteration.

* ``true``: Eliminate dense columns (default)
* ``false``: Don't eliminate dense columns

This setting only has an effect when the ADAT (normal equation) system is solved.

.. note:: the default value is ``true``.

cuDSS Deterministic Mode
"""""""""""""""""""""""""

``CUOPT_CUDSS_DETERMINISTIC`` controls whether cuDSS operates in deterministic mode. Deterministic mode ensures reproducible results across runs but may be slower.

* ``true``: Use deterministic mode
* ``false``: Use non-deterministic mode (default)

.. note:: the default value is ``false``. Enable deterministic mode if reproducibility is more important than performance.

Dual Initial Point
""""""""""""""""""

``CUOPT_BARRIER_DUAL_INITIAL_POINT`` controls the method used to compute the dual initial point for the barrier solver. The choice of initial point will affect the number of iterations performed by barrier.

* ``-1``: Automatic (default) - cuOpt selects the best method
* ``0``: Use an initial point from a heuristic approach based on the paper "On Implementing Mehrotra's Predictor–Corrector Interior-Point Method for Linear Programming" (SIAM J. Optimization, 1992) by Lustig, Martsten, Shanno.
* ``1``: Use an initial point from solving a least squares problem that minimizes the norms of the dual variables and reduced costs while statisfying the dual equality constraints.

.. note:: the default value is ``-1`` (automatic).

Absolute Primal Tolerance
^^^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_ABSOLUTE_PRIMAL_TOLERANCE`` controls the absolute primal tolerance used in the primal feasibility check.
Changing this value might have a significant impact on accuracy and runtime if the relative part
(the right-hand side vector b L2 norm) is close to, or equal to, 0.


The primal feasibility condition is computed as follows::

   primal_feasiblity < absolute_primal_tolerance + relative_primal_tolerance * l2_norm(b)

Default value is ``1e-4``.


Relative Primal Tolerance
^^^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_RELATIVE_PRIMAL_TOLERANCE`` controls the relative primal tolerance used in PDLP's primal feasibility check.
Changing this value has a significant impact on accuracy and runtime.
The primal feasibility condition is computed as follows::

   primal_feasiblity < absolute_primal_tolerance + relative_primal_tolerance * l2_norm(b)

.. note:: the default value is ``1e-4``.

Absolute Dual Tolerance
^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_ABSOLUTE_DUAL_TOLERANCE`` controls the absolute dual tolerance used in PDLP's dual feasibility check.
Changing this value might have a significant impact on accuracy and runtime if the relative part
(the objective vector L2 norm) is close to, or equal to, 0.

The dual feasibility condition is computed as follows::

   dual_feasiblity < absolute_dual_tolerance + relative_dual_tolerance * l2_norm(c)

.. note:: the default value is ``1e-4``.

Relative Dual Tolerance
^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_RELATIVE_DUAL_TOLERANCE`` controls the relative dual tolerance used in PDLP's dual feasibility check.
Changing this value has a significant impact on accuracy and runtime.
The dual feasibility condition is computed as follows::

   dual_feasiblity < absolute_dual_tolerance + relative_dual_tolerance * l2_norm(c)

.. note:: the default value is ``1e-4``.


Absolute Gap Tolerance
^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_ABSOLUTE_GAP_TOLERANCE`` controls the absolute gap tolerance used in PDLP's duality gap check.
Changing this value might have a significant impact on accuracy and runtime if the relative part ``(|primal_objective| + |dual_objective|)`` is close to, or equal to, 0.

The duality gap is computed as follows::

   duality_gap < absolute_gap_tolerance + relative_gap_tolerance * (|primal_objective| + |dual_objective|)

.. note:: the default value is ``1e-4``.


Relative Gap Tolerance
^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_RELATIVE_GAP_TOLERANCE`` controls the relative gap tolerance used in PDLP's duality gap check.
Changing this value has a significant impact on accuracy and runtime.
The duality gap is computed as follows::

   duality_gap < absolute_gap_tolerance + relative_gap_tolerance * (|primal_objective| + |dual_objective|)

.. note:: the default value is ``1e-4``.


Mixed Integer Linear Programming
---------------------------------

We now describe parameter settings for the MILP solvers


Heuristics only
^^^^^^^^^^^^^^^

``CUOPT_MIP_HEURISTICS_ONLY`` controls if only the GPU heuristics should be run for the MIP problem. When set to true, only the primal
bound is improved via the GPU. When set to false, both the GPU and CPU are used and
the dual bound is improved on the CPU.

.. note:: the default value is false.

Scaling
^^^^^^^

``CUOPT_MIP_SCALING`` controls if scaling should be applied to the MIP problem. When true scaling is applied,
when false, no scaling is applied.

.. note:: the defaulte value is true.


Absolute Tolerance
^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_ABSOLUTE_TOLERANCE`` controls the MIP absolute tolerance.

.. note:: the default value is ``1e-6``.

Relative Tolerance
^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_RELATIVE_TOLERANCE`` controls the MIP relative tolerance.

.. note:: the default value is ``1e-12``.


Integrality Tolerance
^^^^^^^^^^^^^^^^^^^^^

``CUOPT_INTEGRALITY_TOLERANCE`` controls the MIP integrality tolerance. A variable is considered to be integral, if
it is within the integrality tolerance of an integer.

.. note:: the default value is ``1e-5``.

Absolute MIP Gap
^^^^^^^^^^^^^^^^

``CUOPT_MIP_ABSOLUTE_GAP`` controls the absolute tolerance used to terminate the MIP solve. The solve terminates when::

    Best Objective - Dual Bound  <= absolute tolerance

when minimizing or

    Dual Bound - Best Objective <= absolute tolerance

when maximizing.

.. note:: the default value is ``1e-10``.

Relative MIP Gap
^^^^^^^^^^^^^^^^

``CUOPT_MIP_RELATIVE_GAP`` controls the relative tolerance used to terminate the MIP solve. The solve terminates when::

    abs(Best Objective - Dual Bound) / abs(Best Objective) <= relative tolerance

If the Best Objective and the Dual Bound are both zero the gap is zero. If the best objective value is zero, the
gap is infinity.

.. note:: the default value is ``1e-4``.
