============================
Convex Optimization Examples
============================

These examples show the Java modeling patterns corresponding to the Python
LP/QP examples. They assume the Java module has been compiled as described in
:doc:`../quick-start` and that the application can load ``libcuopt_jni``.

Simple linear programming
--------------------------

The high-level API uses fluent expressions and explicit comparison methods.

.. code-block:: java

   import com.nvidia.cuopt.linearprogramming.*;

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

``Problem.solve`` copies the model to a native ``DataModel`` and populates
the ``Variable`` and ``Constraint`` objects after the solve. The solution
object remains available for detailed native results and statistics.

Low-level CSR linear program
-----------------------------

Use ``DataModel`` when the input is already in CSR form:

.. code-block:: java

   CsrMatrix matrix = new CsrMatrix(
       new int[] {0, 2},       // row offsets
       new int[] {0, 1},       // column indices
       new double[] {1.0, 1.0});

   try (DataModel model = DataModel.createProblem(
          1, 2,
          ObjectiveSense.MINIMIZE,
          0.0,
          new double[] {1.0, 1.0},
          matrix,
          new byte[] {(byte) 'G'},
          new double[] {10.0},
          new double[] {0.0, 0.0},
          new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY},
          new byte[] {(byte) 'C', (byte) 'C'});
        SolverSettings settings = new SolverSettings().setMethod(SolverMethod.PDLP);
        Solution solution = model.solve(settings)) {
     System.out.println(solution.getPrimalObjective());
     System.out.println(model.getConstraintMatrix().getRowOffsets().length);
   }

For ranged rows, use ``createRangedProblem`` with
``constraintLowerBounds`` and ``constraintUpperBounds`` instead of row sense
and RHS arrays. The mutable setters provide the same representation after an
empty ``new DataModel()``.

Simple quadratic programming
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
       System.out.println("LP stats gap = " + solution.getLpStats().getGap());
     }
   }

The lower-level equivalent is ``DataModel.setQuadraticObjective`` with a
``QuadraticExpression`` or ``setQuadraticObjectiveMatrix`` with quadratic CSR
arrays. For QP solutions, ``getPrimalSolution``, ``getDualSolution``,
``getReducedCost``, ``getDualObjective``, and ``getLpStats`` are available when
the solver returns the corresponding values.

Quadratic constraints
---------------------

Quadratic constraints can be added to a high-level ``Problem`` or directly to
a ``DataModel``:

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

Only ``LE`` and ``GE`` quadratic constraints are supported. Calling
``QuadraticExpression.eq`` or adding an equality quadratic constraint raises
``IllegalArgumentException``.

Reading and writing MPS/QPS
---------------------------

``DataModel`` and ``Problem`` expose both extension-dispatch and direct MPS
entry points:

.. code-block:: java

   try (DataModel model = DataModel.read("model.mps")) {
     System.out.println("Variables: " + model.getNumVariables());
     System.out.println("QP terms: " + model.getQuadraticObjectiveValues().length);
     model.writeMPS("roundtrip.mps");
   }

   try (DataModel fixed = DataModel.parseMps("fixed-format.mps", true)) {
     // Use fixed-format parsing explicitly.
   }

``Problem.read`` and ``Problem.readMPS`` build the high-level Java model from
the parsed data. Parsing failures are reported as ``CuOptException`` with the
cuOpt status code available from ``getStatusCode``.

Inspecting solutions and PDLP warm starts
------------------------------------------

LP solutions expose residuals and solver metadata through ``LPStats``. When
PDLP warm-start data is available, retrieve a defensive-copy representation:

.. code-block:: java

   try (SolverSettings settings = new SolverSettings().setMethod(SolverMethod.PDLP);
        Solution solution = problem.solve(settings)) {
     if (solution.hasPdlpWarmStartData()) {
       PDLPWarmStartData warmStart = solution.getPdlpWarmStartData();
       System.out.println(warmStart.getCurrentPrimalSolution().length);
       System.out.println(warmStart.getTotalPdlpIterations());
     }
   }

The same ``PDLPWarmStartData`` can be supplied to
``SolverSettings.setPdlpWarmStartData`` for a subsequent compatible LP solve.
