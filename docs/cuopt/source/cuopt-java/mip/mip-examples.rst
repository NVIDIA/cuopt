============
MIP Examples
============

These examples show mixed-integer modeling, MIP starts, semi-continuous
variables, and incumbent callbacks in Java.

Simple MILP
-----------

.. code-block:: java

   import com.nvidia.cuopt.linearprogramming.*;

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
              .setParameter(CuOptConstants.CUOPT_TIME_LIMIT, 10.0);
          Solution solution = problem.solve(settings)) {
       System.out.println("Status: " + solution.getTerminationStatus());
       System.out.println("x = " + x.getValue());
       System.out.println("y = " + y.getValue());
       System.out.println("Objective = " + solution.getPrimalObjective());
       System.out.println("MIP gap = " + solution.getMipGap());
       System.out.println("Bound = " + solution.getSolutionBound());
       System.out.println("Nodes = " + solution.getMipStats().getNumNodes());
     }
   }

The MIP solver can return a feasible solution before proving optimality. Use
the termination status, MIP gap, and solution bound together when interpreting
the result.

Semi-continuous variables
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

MIP starts
----------

Set starts on variables when using the high-level ``Problem`` API:

.. code-block:: java

   x.setMipStart(3.0);
   y.setMipStart(2.0);

   try (SolverSettings settings = new SolverSettings();
        Solution solution = problem.solve(settings)) {
     System.out.println(solution.getPrimalObjective());
   }

For a lower-level model, pass a full variable-index-ordered array through
``SolverSettings.addMipStart``.

Incumbent callback
------------------

Register an incumbent callback before solving:

.. code-block:: java

   try (SolverSettings settings = new SolverSettings()) {
     settings.setMipCallback(
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

The callback receives a defensive Java array containing the incumbent vector,
the incumbent objective, the current solution bound, and the user data object.

LP relaxation
-------------

Create a continuous relaxation without changing the original MIP:

.. code-block:: java

   try (Problem relaxed = problem.relax();
        Solution solution = relaxed.solve()) {
     System.out.println("LP relaxation objective = " + solution.getPrimalObjective());
   }
