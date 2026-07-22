============
MIP Settings
============


This page describes the parameter settings available for cuOpt's MIP solver. These parameters are set as :ref:`parameter constants <parameter-constants>` in case of C API and in case of Server Thin client as raw strings. Refer to examples in :doc:`C </cuopt-c/mip/index>` and :doc:`Server Thin client </cuopt-server/index>` for more details.

.. note::
   When setting parameters in thin client solver settings, remove ``CUOPT_`` from the parameter name and convert to lowercase. For example, ``CUOPT_TIME_LIMIT`` would be set as ``time_limit``.

Common Parameters
-----------------

We begin by describing parameters common to both the MILP and LP solvers.


Time Limit
^^^^^^^^^^
``CUOPT_TIME_LIMIT`` controls the time limit in seconds after which the solver will stop and return the current solution.
For performance reasons, cuOpt does not constantly checks for time limit. Thus, the solver
may run slightly over the limit. If set along with the iteration limit, cuOpt will stop when
the first limit (iteration or time) is hit.


.. note:: By default there is no time limit. So cuOpt will run until it finds an optimal solution,
   or proves the problem is infeasible or unbounded.



Log to Console
^^^^^^^^^^^^^^
``CUOPT_LOG_TO_CONSOLE`` controls whether cuOpt should log information to the console during a solve.
If true, a logging info is written to the console; if false no logging info is written to the console (logs may still be written to a file.)

.. note:: The default value is true.

Log File
^^^^^^^^
``CUOPT_LOG_FILE`` controls the name of a log file where cuOpt should write information about the solve.

.. note:: The default value is ``""`` and no log file is written. This setting is ignored by the cuOpt service, use the log callback feature instead.

Solution File
^^^^^^^^^^^^^
``CUOPT_SOLUTION_FILE`` controls the name of a file where cuOpt should write the solution.

.. note:: The default value is ``""`` and no solution file is written. This setting is ignored by the cuOpt service.

User Problem File
^^^^^^^^^^^^^^^^^
``CUOPT_USER_PROBLEM_FILE`` controls the name of a file where cuOpt should write the user problem.

.. note:: The default value is ``""`` and no user problem file is written. This setting is ignored by the cuOpt service.

Num CPU Threads
^^^^^^^^^^^^^^^
``CUOPT_NUM_CPU_THREADS`` controls the number of CPU threads used in the MIP solver. Set this to a small value to limit
the amount of CPU resources cuOpt uses. Set this to a large value to improve solve times for CPU
parallel parts of the solvers.

.. note:: By default the number of CPU threads is automatically determined based on the number of CPU cores.

Presolve
^^^^^^^^
``CUOPT_PRESOLVE`` controls whether to apply presolve reductions. Set this to 0 to disable presolve.

.. note:: By default, presolve is enabled.

Probing
^^^^^^^
``CUOPT_MIP_PROBING`` toggles the probing-cache step of cuOpt's internal MIP presolve. The probing pass evaluates variable fixings to discover implications used later by branch-and-bound and the rounding heuristics. It is enabled by default (``true``); set it to ``false`` to skip probing while leaving the rest of presolve in place. Setting ``CUOPT_PRESOLVE=0`` already turns off the entire presolve pipeline, so ``CUOPT_MIP_PROBING`` only matters when presolve is otherwise enabled. Probing is also skipped in deterministic mode and on LP-only solves.

Dual Postsolve
^^^^^^^^^^^^^^
``CUOPT_DUAL_POSTSOLVE`` controls whether dual postsolve is enabled when using Papilo presolver for LP problems. Disabling dual postsolve can improve solve time at the expense of not having
access to the dual solution. Enabled by default for LP when Papilo presolve is selected. This is not relevant for MIP problems.

Mixed Integer Linear Programming
---------------------------------

The cuOpt MIP solver is in **beta** and under active development. The solver
currently excels at finding high-quality feasible solutions quickly with
GPU-accelerated primal heuristics. Proving feasible solutions optimal remains
under active development.

We now describe parameter settings for the MILP solver.


Heuristics Only
^^^^^^^^^^^^^^^

``CUOPT_MIP_HEURISTICS_ONLY`` controls if only the GPU heuristics should be run for the MIP problem. When set to true, only the primal
bound is improved via the GPU. When set to false, both the GPU and CPU are used and
the dual bound is improved on the CPU.

.. note:: The default value is false.

Scaling
^^^^^^^

``CUOPT_MIP_SCALING`` controls if scaling should be applied to the MIP problem.

* ``0``: Scaling is off.
* ``1``: Scaling is on.
* ``2``: Scaling is not applied to the objective (default).

Absolute Tolerance
^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_ABSOLUTE_TOLERANCE`` controls the MIP absolute tolerance.

.. note:: The default value is ``1e-6``.

Relative Tolerance
^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_RELATIVE_TOLERANCE`` controls the MIP relative tolerance.

.. note:: The default value is ``1e-12``.


Integrality Tolerance
^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_INTEGRALITY_TOLERANCE`` controls the MIP integrality tolerance. A variable is considered to be integral, if
it is within the integrality tolerance of an integer.

.. note:: The default value is ``1e-5``.

Absolute MIP Gap
^^^^^^^^^^^^^^^^

``CUOPT_MIP_ABSOLUTE_GAP`` controls the absolute tolerance used to terminate the MIP solve. The solve terminates when::

    Best Objective - Dual Bound  <= absolute tolerance

when minimizing or

    Dual Bound - Best Objective <= absolute tolerance

when maximizing.

.. note:: The default value is ``1e-10``.

Relative MIP Gap
^^^^^^^^^^^^^^^^

``CUOPT_MIP_RELATIVE_GAP`` controls the relative tolerance used to terminate the MIP solve. The solve terminates when::

    abs(Best Objective - Dual Bound) / abs(Best Objective) <= relative tolerance

If the Best Objective and the Dual Bound are both zero the gap is zero. If the best objective value is zero, the
gap is infinity.

.. note:: The default value is ``1e-4``.


Node Limit
^^^^^^^^^^

``CUOPT_NODE_LIMIT`` controls the maximum number of branch-and-bound nodes the MILP solver will explore before stopping and returning the current best feasible solution (if any). If set along with the time limit, cuOpt stops at whichever limit is hit first.

.. note:: By default there is no node limit.

Cut Passes
^^^^^^^^^^

``CUOPT_MIP_CUT_PASSES`` controls the maximum number of cut passes to run. Set this value to 0 to disable cuts. Set this value to larger numbers to perform more cut passes.

.. note:: The default value is ``10``.

Mixed Integer Rounding Cuts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_MIXED_INTEGER_ROUNDING_CUTS`` controls whether to use mixed integer rounding cuts.
The default value of ``-1`` (automatic) means that the solver will decide whether to use mixed integer rounding cuts based on the problem characteristics.
Set this value to 1 to enable mixed integer rounding cuts.
Set this value to 0 to disable mixed integer rounding cuts.

.. note:: The default value is ``-1`` (automatic).

Mixed Integer Gomory Cuts
^^^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_MIXED_INTEGER_GOMORY_CUTS`` controls whether to use mixed integer Gomory cuts.
The default value of ``-1`` (automatic) means that the solver will decide whether to use mixed integer Gomory cuts based on the problem characteristics.
Set this value to 1 to enable mixed integer Gomory cuts.
Set this value to 0 to disable mixed integer Gomory cuts.

.. note:: The default value is ``-1`` (automatic).

Strong Chvatal-Gomory Cuts
^^^^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_STRONG_CHVATAL_GOMORY_CUTS`` controls whether to use strong Chvatal-Gomory cuts.
The default value of ``-1`` (automatic) means that the solver will decide whether to use strong Chvatal-Gomory cuts based on the problem characteristics.
Set this value to 1 to enable strong Chvatal Gomory cuts.
Set this value to 0 to disable strong Chvatal Gomory cuts.

.. note:: The default value is ``-1`` (automatic).

Knapsack Cuts
^^^^^^^^^^^^^

``CUOPT_MIP_KNAPSACK_CUTS`` controls whether to use knapsack cuts.
The default value of ``-1`` (automatic) means that the solver will decide whether to use knapsack cuts based on the problem characteristics.
Set this value to 1 to enable knapsack cuts.
Set this value to 0 to disable knapsack cuts.

.. note:: The default value is ``-1`` (automatic).


Cut Change Threshold
^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_CUT_CHANGE_THRESHOLD`` controls the threshold for the improvement in the dual bound per cut pass.
Larger values require the dual bound to improve significantly in each cut pass.
Set this value to -1 to allow the cut passes to continue even if the dual bound does not improve.

.. note:: The default value is ``-1`` (no threshold).

Cut Min Orthogonality
^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_CUT_MIN_ORTHOGONALITY`` controls the minimum orthogonality required for a cut to be added to the LP relaxation.
Set this value close to 1, to require all cuts be close to orthogonal to each other.
Set this value close to zero to allow more cuts to be added to the LP relaxation.

.. note:: The default value is ``0.5``.

Reduced Cost Strengthening
^^^^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_REDUCED_COST_STRENGTHENING`` controls whether to use reduced-cost strengthening.
When enabled, the solver will use integer feasible solutions to strengthen the bounds of integer variables.
The default value of ``-1`` (automatic) means that the solver will decide whether to use reduced-cost strengthening based on the problem characteristics.
Set this value to 0 to disable reduced-cost strengthening.
Set this value to 1 to perform reduced-cost strengthening during the root cut passes.
Set this value to 2 to perform reduced-cost strengthening during the root cut passes and after strong branching.

.. note:: The default value is ``-1`` (automatic).

Reliability Branching
^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_RELIABILITY_BRANCHING`` controls the reliability branching mode.
The default value of ``-1`` (automatic) means that the solver will decide whether to use reliability branching, and the reliability branching factor, based on the problem characteristics.
Set this value to 0 to disable reliability branching.
Set this value to k > 0, to enable reliability branching. A variable will be considered reliable if it has been branched on k times.

.. note:: The default value is ``-1`` (automatic).

Batch PDLP Strong Branching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_BATCH_PDLP_STRONG_BRANCHING`` controls whether to use batched PDLP over Dual Simplex during strong branching at the root.
When enabled, the solver evaluates multiple branching candidates simultaneously in a single batched PDLP solve rather than solving them in parallel using Dual Simplex. This can significantly reduce the time spent in strong branching if Dual Simplex is struggling.
Set this value to 0 to disable batched PDLP strong branching.
Set this value to 1 to enable batched PDLP strong branching.

.. note:: The default value is ``0`` (disabled). This setting is ignored if the problem is not a MIP problem.

Batch PDLP Reliability Branching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_BATCH_PDLP_RELIABILITY_BRANCHING`` controls whether to use batched PDLP for reliability branching evaluations.
When enabled, candidate variables for reliability branching are evaluated simultaneously using a single batched PDLP solve.
Set this value to 0 to disable.
Set this value to 1 to enable.

.. note:: The default value is ``0`` (disabled).

Strong Branching Simplex Iteration Limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_STRONG_BRANCHING_SIMPLEX_ITERATION_LIMIT`` controls the maximum number of simplex iterations allowed per candidate during strong branching.
Reducing this value speeds up strong branching at the cost of less accurate candidate evaluations.

.. note:: The default value is ``-1`` (choose the iteration limit automatically).

MIP Determinism Mode
^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_DETERMINISM_MODE`` controls whether the MIP solver runs in opportunistic or deterministic mode.

* ``0`` (``CUOPT_MODE_OPPORTUNISTIC``): Opportunistic mode — results may vary between runs due to parallelism (default).
* ``1`` (``CUOPT_MODE_DETERMINISTIC``): Deterministic mode — improves reproducibility across runs with the same number of threads.

.. note:: The default value is ``0`` (opportunistic).

.. warning:: Deterministic mode is experimental. It improves reproducibility in many cases but does not yet guarantee fully deterministic results in all scenarios.

MIP Symmetry
^^^^^^^^^^^^^

``CUOPT_MIP_SYMMETRY`` controls symmetry detection and handling in the MIP solver.

* ``-1``: Automatic (default) — cuOpt decides based on problem characteristics.
* ``0``: Disable symmetry handling.
* ``1``: Enable symmetry handling (orbital fixing only).
* ``2``: Enable symmetry handling (orbital fixing + lexical reduction).

.. note:: The default value is ``-1`` (automatic).

Flow Cover Cuts
^^^^^^^^^^^^^^^^

``CUOPT_MIP_FLOW_COVER_CUTS`` controls whether to use flow cover cuts.
Set this value to ``-1`` (automatic) to let the solver decide.
Set this value to ``0`` to disable flow cover cuts.
Set this value to ``1`` to enable flow cover cuts.

.. note:: The default value is ``-1`` (automatic).

Implied Bound Cuts
^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_IMPLIED_BOUND_CUTS`` controls whether to use implied bound cuts.
Set this value to ``-1`` (automatic) to let the solver decide.
Set this value to ``0`` to disable implied bound cuts.
Set this value to ``1`` to enable implied bound cuts.

.. note:: The default value is ``-1`` (automatic).

Clique Cuts
^^^^^^^^^^^^

``CUOPT_MIP_CLIQUE_CUTS`` controls whether to use clique cuts.
Set this value to ``-1`` (automatic) to let the solver decide.
Set this value to ``0`` to disable clique cuts.
Set this value to ``1`` to enable clique cuts.

.. note:: The default value is ``-1`` (automatic).

Zero-Half Cuts
^^^^^^^^^^^^^^

``CUOPT_MIP_ZERO_HALF_CUTS`` controls whether to use zero-half cuts.
Set this value to ``-1`` (automatic) to let the solver decide.
Set this value to ``0`` to disable zero-half cuts.
Set this value to ``1`` to enable zero-half cuts.

.. note:: The default value is ``-1`` (automatic).

Objective Step
^^^^^^^^^^^^^^^

``CUOPT_MIP_OBJECTIVE_STEP`` controls whether cuOpt automatically detects and exploits discrete step structure in the objective to tighten the dual bound.

* ``0``: Disable objective step detection.
* ``1``: Enable objective step detection (default).

.. note:: The default value is ``1`` (enabled).

Semi-Continuous Big-M
^^^^^^^^^^^^^^^^^^^^^^

``CUOPT_MIP_SEMICONTINUOUS_BIG_M`` controls the Big-M coefficient used when linearizing semi-continuous variable constraints.
A semi-continuous variable is either zero or takes a value in the range ``[lower_bound, upper_bound]``.
Set this to a positive value that is at least as large as the upper bound of any semi-continuous variable in the problem.

.. note:: By default cuOpt derives the Big-M from the variable's upper bound.

Work Limit
^^^^^^^^^^

``CUOPT_WORK_LIMIT`` controls the work limit after which the solver will stop and return the current solution.
Work units are a machine-independent measure of solver effort. If set along with the time limit or iteration limit, cuOpt will stop when the first limit is hit.

.. note:: By default there is no work limit.

Random Seed
^^^^^^^^^^^^

``CUOPT_RANDOM_SEED`` controls the random seed used by the solver. Setting a fixed seed enables reproducible results when running in deterministic mode.

.. note:: By default the random seed is set automatically.

Presolve File
^^^^^^^^^^^^^^

``CUOPT_PRESOLVE_FILE`` controls the name of a file where cuOpt should write presolve information.

.. note:: The default value is ``""`` and no presolve file is written.
