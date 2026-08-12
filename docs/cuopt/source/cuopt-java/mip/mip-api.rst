=================
MIP API Reference
=================

MIP uses the shared Java problem construction and solve APIs documented in
:doc:`../convex/convex-api`. The following features are particularly relevant
to mixed-integer problems.

Variable Types
--------------

Use ``VariableType`` when adding a variable or updating an existing variable:

.. code-block:: java

   Variable integer = problem.addVariable(
       0.0, 100.0, 3.0,
       VariableType.INTEGER, "integer");

   Variable semiContinuous = problem.addVariable(
       0.0, 100.0, 1.0,
       VariableType.SEMI_CONTINUOUS, "semi");

The supported values are ``CONTINUOUS``, ``INTEGER``, and
``SEMI_CONTINUOUS``. ``Problem.isMIP()`` and ``Solution.isMIP()`` report
whether a problem or result contains a noncontinuous variable.

MIP Starts
----------

MIP starts can be provided per variable through ``Variable.setMIPStart``. The
high-level ``Problem.solve`` collects defined variable starts and passes them
to the native solver. A complete start can also be supplied directly through
``SolverSettings.addMIPStart(double[])``.

.. code-block:: java

   x.setMIPStart(3.0);
   y.setMIPStart(2.0);

   try (SolverSettings settings = new SolverSettings()) {
     // The array follows the problem's variable-index order.
     settings.addMIPStart(new double[] {3.0, 2.0});
     try (Solution solution = problem.solve(settings)) {
       System.out.println(solution.getMIPGap());
     }
   }

MIP Settings
------------

All solver settings are set through ``SolverSettings``. Use the overloaded
``setSetting`` methods for string, integer, floating-point, and boolean
values. MIP-relevant settings include time and node limits, MIP tolerances,
presolve, heuristics, scaling, determinism, and cut controls. The generated
``CuOptConstants`` class contains the string and integer constants from the
cuOpt public constants header, including every setting name.

MIP Callbacks
-------------

``SolverSettings.setMIPCallback`` accepts either callback interface:

``MIPSolutionCallback`` receives each incumbent solution:

.. code-block:: java

   settings.setMIPCallback(
       (solution, objectiveValue, solutionBound, userData) -> {
         System.out.println("incumbent objective = " + objectiveValue);
       },
       "my-user-data",
       problem.getNumVariables());

``MIPSetSolutionCallback`` returns a candidate solution and objective when the
native solver asks Java for one:

.. code-block:: java

   settings.setMIPCallback(
       (solutionBound, userData) ->
           new MIPCallbackSolution(new double[] {3.0, 2.0}, 19.0),
       null,
       problem.getNumVariables());

Callbacks are native-runtime features. Keep the callback and any user data
valid for the duration of the solve, and close the ``SolverSettings`` after the
solve completes. Registered callbacks can be inspected with
``getMIPCallbacks``.

MIP Solution Fields
-------------------

For a MIP ``Solution``:

* ``getPrimalSolution`` returns the incumbent primal vector;
* ``getPrimalObjective`` returns its objective value;
* ``getMIPGap`` returns the current relative MIP gap;
* ``getSolutionBound`` returns the best bound reported by the solver;
* ``getMIPStats`` returns ``MIPStats``; and
* ``getTerminationStatus``, ``getTerminationReason``, ``getErrorStatus``,
  ``getErrorMessage``, and ``getSolveTime`` describe the solve.

``MIPStats`` contains presolve time, maximum constraint violation, maximum
integer violation, maximum variable-bound violation, node count, and simplex
iteration count. LP-only accessors such as ``getDualSolution``,
``getReducedCost``, and ``getLPStats`` raise ``IllegalStateException`` for a
MIP result.

Relaxing and Inspecting a MIP
-----------------------------

``Problem.relax()`` returns a separate continuous problem while preserving
variable names, bounds, objective, and constraints. This is useful for
inspecting the LP relaxation without changing the original MIP. The original
problem can also be inspected through ``getCSR``, ``getQCSR``, and
``getIncumbentValues``.
