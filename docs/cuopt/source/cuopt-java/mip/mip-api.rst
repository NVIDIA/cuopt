=================
MIP API Reference
=================

MIP uses the shared Java modeling and solve APIs documented in
:doc:`../convex/convex-api`. The following features are particularly relevant
to mixed-integer models.

Variable types
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
``SEMI_CONTINUOUS``. ``Problem.isMip()``, ``DataModel.isMip()``, and
``Solution.isMip()`` report whether a model or result contains a noncontinuous
variable.

MIP starts
----------

MIP starts can be provided per variable through ``Variable.setMipStart``. The
high-level ``Problem.solve`` collects defined variable starts and passes them
to the native solver. A complete start can also be supplied directly through
``SolverSettings.addMipStart(double[])``.

.. code-block:: java

   x.setMipStart(3.0);
   y.setMipStart(2.0);

   try (SolverSettings settings = new SolverSettings()) {
     // The array follows the model's variable-index order.
     settings.addMipStart(new double[] {3.0, 2.0});
     try (Solution solution = problem.solve(settings)) {
       System.out.println(solution.getMipGap());
     }
   }

MIP settings
------------

All solver parameters are set through ``SolverSettings``. Use the overloaded
``setParameter`` methods for string, integer, floating-point, and boolean
values. MIP-relevant settings include time and node limits, MIP tolerances,
presolve, heuristics, scaling, determinism, and cut controls. Parameter names
are available at runtime through ``SolverSettings.getSolverParameterNames``;
the generated ``CuOptConstants`` class contains string and integer constants
from the cuOpt public constants header.

MIP callbacks
-------------

``SolverSettings.setMipCallback`` accepts either callback interface:

``MipSolutionCallback`` receives each incumbent solution:

.. code-block:: java

   settings.setMipCallback(
       (solution, objectiveValue, solutionBound, userData) -> {
         System.out.println("incumbent objective = " + objectiveValue);
       },
       "my-user-data",
       problem.getNumVariables());

``MipSetSolutionCallback`` returns a candidate solution and objective when the
native solver asks Java for one:

.. code-block:: java

   settings.setMipCallback(
       (solutionBound, userData) ->
           new MipCallbackSolution(new double[] {3.0, 2.0}, 19.0),
       null,
       problem.getNumVariables());

Callbacks are native-runtime features. Keep the callback and any user data
valid for the duration of the solve, and close the ``SolverSettings`` after the
solve completes. Registered callbacks can be inspected with
``getMipCallbacks``.

MIP solution fields
-------------------

For a MIP ``Solution``:

* ``getPrimalSolution`` returns the incumbent primal vector;
* ``getPrimalObjective`` returns its objective value;
* ``getMipGap`` returns the current relative MIP gap;
* ``getSolutionBound`` returns the best bound reported by the solver;
* ``getMipStats`` returns ``MIPStats``; and
* ``getTerminationStatus``, ``getTerminationReason``, ``getErrorStatus``,
  ``getErrorMessage``, and ``getSolveTime`` describe the solve.

``MIPStats`` contains presolve time, maximum constraint violation, maximum
integer violation, maximum variable-bound violation, node count, and simplex
iteration count. LP-only accessors such as ``getDualSolution``,
``getReducedCost``, and ``getLpStats`` raise ``IllegalStateException`` for a
MIP result.

Relaxing and inspecting a MIP
-----------------------------

``Problem.relax()`` returns a separate continuous model while preserving
variable names, bounds, objective, and constraints. This is useful for
inspecting the LP relaxation without changing the original MIP. The original
model can also be inspected through ``getCSR``, ``getQCSR``, ``toDataModel``,
and ``getIncumbentValues``.
