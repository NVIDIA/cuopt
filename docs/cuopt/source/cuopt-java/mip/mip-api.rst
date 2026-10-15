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

Passing a complete start directly means building the array yourself. It is
indexed by ``Variable.getIndex()``, which is the order ``getVariables``
returns, so build it from that list rather than by hand:

.. code-block:: java

   Map<Variable, Double> start = Map.of(x, 3.0, y, 2.0);

   double[] values = new double[problem.getNumVariables()];
   for (Variable variable : problem.getVariables()) {
     values[variable.getIndex()] = start.getOrDefault(variable, 0.0);
   }

   try (SolverSettings settings = new SolverSettings()) {
     settings.addMIPStart(values);
     try (Solution solution = problem.solve(settings)) {
       System.out.println(solution.getMIPGap());
     }
   }

MIP Settings
------------

All solver settings are set through ``SolverSettings``. Use the overloaded
``setSetting`` methods for string, integer, floating-point, and Boolean
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

``MIPSetSolutionCallback`` runs in the other direction: the solver asks your
code for a solution to try, and the ``MIPCallbackSolution`` you return carries
it back. Use it to feed in a solution found elsewhere — a heuristic of your
own, or a result carried over from a previous solve. Returning ``null``
declines, leaving the search untouched.

The array is in variable-index order and must cover every variable, so build it
the same way as a MIP start:

.. code-block:: java

   settings.setMIPCallback(
       (solutionBound, userData) -> {
         double[] values = new double[problem.getNumVariables()];
         for (Variable variable : problem.getVariables()) {
           values[variable.getIndex()] = myHeuristic(variable);
         }
         return new MIPCallbackSolution(values, objectiveOf(values));
       },
       null,
       problem.getNumVariables());

Callbacks are native-runtime features. Keep the callback and any user data
valid for the duration of the solve, and close the ``SolverSettings`` after the
solve completes. Registered callbacks can be inspected with
``getMIPCallbacks``.

MIP Solution Fields
-------------------

For a MIP ``Solution``:

* ``getPrimalObjective`` returns the incumbent objective value;
* ``getMIPGap`` returns the current relative MIP gap;
* ``getSolutionBound`` returns the best bound reported by the solver; and
* ``getTerminationStatus``, ``getErrorStatus``, and ``getErrorMessage``
  describe the solve.

Incumbent values are read from the model: ``Variable.getValue`` after the solve.

LP-only accessors such as ``getDualSolution`` and ``getReducedCost`` raise
``IllegalStateException`` for a MIP result.

MIP statistics — presolve time, node and simplex iteration counts, and the
violation magnitudes — are read as solution attributes:

.. code-block:: java

   int nodes = solution.getIntAttribute(
       CuOptConstants.CUOPT_SOLUTION_ATTR_MIP_NUM_NODES);
   double violation = solution.getFloatAttribute(
       CuOptConstants.CUOPT_SOLUTION_ATTR_MIP_MAX_CONSTRAINT_VIOLATION);

An LP selector on a MIP solution raises ``CuOptException``.

Inspecting a MIP
----------------

A problem can be inspected through ``getConstraintMatrix`` and
``getQuadraticObjectiveMatrix``. To create an LP relaxation, build the problem
with ``VariableType.CONTINUOUS``, or set the types through
``Variable.setVariableType`` before solving.
