============================
Convex Optimization Examples
============================

These examples show the Java modeling patterns for LP and QP problems. They
assume the Java module has been compiled as described in
:doc:`../quick-start` and that the application can load ``libcuopt_jni``.

Simple Linear Programming
--------------------------

The high-level API uses fluent expressions and explicit comparison methods.

.. code-block:: java

   import com.nvidia.cuopt.mathematicaloptimization.*;

   try (Problem problem = new Problem("simple-lp")) {
     Variable x = problem.addVariable(
         0.0, Double.POSITIVE_INFINITY, 1.0,
         VariableType.CONTINUOUS, "x");
     Variable y = problem.addVariable(
         0.0, Double.POSITIVE_INFINITY, 1.0,
         VariableType.CONTINUOUS, "y");

     problem.addConstraint(
         LinearExpression.of(x).plus(y).ge(10.0), "demand");
     problem.setObjective(
         LinearExpression.of(x).plus(y), ObjectiveSense.MINIMIZE);

     try (SolverSettings settings = new SolverSettings()
              .setMethod(SolverMethod.PDLP);
          Solution solution = problem.solve(settings)) {
       System.out.println("Status: " + solution.getTerminationStatus());
       System.out.println("x = " + x.getValue());
       System.out.println("y = " + y.getValue());
       System.out.println("Objective = " + solution.getPrimalObjective());
     }
   }

``Problem.solve`` populates the ``Variable`` and ``Constraint`` objects after
the solve. The solution object remains available for detailed native results
and statistics.

Simple Quadratic Programming
-----------------------------

Quadratic objectives combine quadratic, linear, and constant terms:

.. code-block:: java

   try (Problem problem = new Problem("simple-qp")) {
     Variable x = problem.addVariable(0.0, 10.0, 0.0, VariableType.CONTINUOUS, "x");
     Variable y = problem.addVariable(0.0, 10.0, 0.0, VariableType.CONTINUOUS, "y");

     QuadraticExpression objective = QuadraticExpression
         .of(x, x, 1.0)
         .plus(y, y, 1.0)
         .plus(LinearExpression.of(x).times(-1.0))
         .plus(LinearExpression.of(y).times(-1.0));

     problem.addConstraint(
         LinearExpression.of(x).plus(y).eq(1.0), "sum");
     problem.setObjective(objective, ObjectiveSense.MINIMIZE);

     try (Solution solution = problem.solve()) {
       System.out.println("x = " + x.getValue());
       System.out.println("y = " + y.getValue());
       System.out.println("Objective = " + solution.getPrimalObjective());
     }
   }

For QP solutions, ``getDualObjective`` is available when the solver returns it,
and variable and constraint values are read from the model through
``Variable.getValue``, ``Variable.getReducedCost``, and
``Constraint.getDualValue``.

Quadratic Constraints
---------------------

Quadratic constraints can be added directly to a ``Problem``:

.. code-block:: java

   try (Problem problem = new Problem("quadratic-constraint")) {
     Variable x = problem.addVariable(0.0, 10.0, 1.0, VariableType.CONTINUOUS, "x");
     Variable y = problem.addVariable(0.0, 10.0, 1.0, VariableType.CONTINUOUS, "y");

     QuadraticExpression radius = QuadraticExpression
         .of(x, x, 1.0)
         .plus(y, y, 1.0);
     problem.addConstraint(radius.le(4.0), "radius");
     problem.setObjective(
         LinearExpression.of(x).plus(y), ObjectiveSense.MAXIMIZE);

     try (Solution solution = problem.solve()) {
       System.out.println(solution.getTerminationStatus());
     }
   }

Only ``LE`` and ``GE`` quadratic constraints are supported;
``QuadraticExpression`` does not expose an ``eq`` method.

Reading and Writing MPS/QPS
---------------------------

``Problem`` exposes both extension-dispatch and direct MPS entry points:

.. code-block:: java

   try (Problem problem = Problem.read("problem.mps")) {
     System.out.println("Variables: " + problem.getNumVariables());
     problem.write("roundtrip.mps");
   }

   try (Problem fixed = Problem.read("fixed-format.mps", true)) {
     // Use fixed-format parsing explicitly.
   }

Parsing failures are reported as ``CuOptException`` with the cuOpt status code
available from ``getStatusCode``.
