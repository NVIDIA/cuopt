============
MIP Examples
============

These examples show mixed-integer modeling, MIP starts, semi-continuous
variables, and incumbent callbacks in Java.

Simple MIP
----------

.. code-block:: java

   import com.nvidia.cuopt.mathematicaloptimization.*;

   try (Problem problem = new Problem("simple-milp")) {
     Variable x = problem.addVariable(
         0.0, 100.0, 3.0, VariableType.INTEGER, "x");
     Variable y = problem.addVariable(
         0.0, 100.0, 5.0, VariableType.INTEGER, "y");

     problem.addConstraint(
         LinearExpression.of(x).times(2.0).plus(y).le(8.0), "capacity");
     problem.setObjective(
         LinearExpression.of(x).times(3.0).plus(y, 5.0),
         ObjectiveSense.MAXIMIZE);

     try (SolverSettings settings = new SolverSettings()
              .setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 10.0);
          Solution solution = problem.solve(settings)) {
       System.out.println("Status: " + solution.getTerminationStatus());
       System.out.println("x = " + x.getValue());
       System.out.println("y = " + y.getValue());
       System.out.println("Objective = " + solution.getPrimalObjective());
       System.out.println("MIP gap = " + solution.getMIPGap());
       System.out.println("Bound = " + solution.getSolutionBound());
     }
   }

The MIP solver can return a feasible solution before proving optimality. Use
the termination status, MIP gap, and solution bound together when interpreting
the result.

Semi-Continuous Variables
-------------------------

``SEMI_CONTINUOUS`` variables are zero or lie within their declared bounds.

.. code-block:: java

   try (Problem problem = new Problem("semi-continuous")) {
     Variable production = problem.addVariable(
         10.0, 100.0, 1.0,
         VariableType.SEMI_CONTINUOUS, "production");
     problem.setObjective(production, ObjectiveSense.MINIMIZE);

     try (Solution solution = problem.solve()) {
       System.out.println("production = " + production.getValue());
     }
   }

MIP Starts
----------

Set starts on variables when using the high-level ``Problem`` API:

.. code-block:: java

   x.setMIPStart(3.0);
   y.setMIPStart(2.0);

   try (SolverSettings settings = new SolverSettings();
        Solution solution = problem.solve(settings)) {
     System.out.println(solution.getPrimalObjective());
   }

Setting a start per variable avoids handling the ordering at all, and is the
form to prefer.

``SolverSettings.addMIPStart`` takes a complete array instead, indexed by
``Variable.getIndex()``. It is the way to supply more than one starting point,
since it can be called repeatedly while each ``Variable`` holds a single value.
Build it from ``getVariables`` so the ordering comes from the problem rather
than from you:

.. code-block:: java

   double[] values = new double[problem.getNumVariables()];
   for (Variable variable : problem.getVariables()) {
     values[variable.getIndex()] = startFor(variable);
   }
   settings.addMIPStart(values);

MIP starts are currently unsupported with presolve on.

Incumbent Callback
------------------

Register an incumbent callback before solving:

.. code-block:: java

   try (SolverSettings settings = new SolverSettings()) {
     settings.setMIPCallback(
         (incumbent, objective, bound, userData) -> {
           System.out.println(
               "incumbent objective=" + objective + ", bound=" + bound);
         },
         null,
         problem.getNumVariables());

     try (Solution solution = problem.solve(settings)) {
       System.out.println("Final status: " + solution.getTerminationStatus());
     }
   }

The callback receives a defensive copy of the incumbent vector, the incumbent
objective, the current solution bound, and the user data object.

The vector is in variable-index order. To read specific variables out of it
without depending on that order, pass them to ``Problem.fromIncumbent``, which
returns their values in the order you ask for:

.. code-block:: java

   settings.setMIPCallback(
       (incumbent, objective, bound, userData) -> {
         double[] picked = Problem.fromIncumbent(incumbent, z, y, x);
         System.out.println("z=" + picked[0] + " y=" + picked[1] + " x=" + picked[2]);
       },
       null,
       problem.getNumVariables());

LP Relaxation
-------------

Relax the integer variables before solving:

.. code-block:: java

   for (Variable variable : problem.getVariables()) {
     variable.setVariableType(VariableType.CONTINUOUS);
   }
   try (Solution solution = problem.solve()) {
     System.out.println("LP relaxation objective = " + solution.getPrimalObjective());
   }
