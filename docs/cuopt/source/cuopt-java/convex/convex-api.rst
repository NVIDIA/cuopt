===================================
Convex Optimization API Reference
===================================

The Java LP/QP bindings are in the package
``com.nvidia.cuopt.linearprogramming``. The public API is documented below by
role. Method names are Java names and therefore use fluent methods instead of
Python operator overloads.

High-level model
----------------

``Problem`` is the recommended entry point for models built in Java.

.. list-table:: ``Problem``
   :header-rows: 1
   :widths: 28 72

   * - API
     - Description
   * - ``new Problem()`` / ``new Problem(String name)``
     - Create an empty model, optionally with a problem name.
   * - ``addVariable(...)``
     - Add a variable with lower/upper bounds, objective coefficient, variable type, and name.
   * - ``addConstraint(Constraint, String name)``
     - Add a linear or quadratic constraint.
   * - ``setObjective(LinearExpression, ObjectiveSense)``
     - Set a linear objective.
   * - ``setObjective(QuadraticExpression, ObjectiveSense)``
     - Set a quadratic objective with optional linear and constant terms.
   * - ``solve()`` / ``solve(SolverSettings)``
     - Convert the model to a native ``DataModel`` and return a ``Solution``.
   * - ``toDataModel()``
     - Materialize the high-level model as the lower-level native data model.
   * - ``getCSR()`` / ``getQCSR()``
     - Inspect the linear or quadratic objective matrix in CSR form.
   * - ``writeMPS(String)`` / ``read(String)`` / ``readMPS(String)``
     - Write or load MPS/QPS-backed models. The fixed-format overloads accept a boolean flag.
   * - ``update()`` / ``updateConstraint(...)`` / ``updateObjective(...)``
     - Update model state and reset solved values where appropriate.
   * - ``relax()``
     - Return a copy with variables converted to continuous type.

The model also exposes ``getVariables``, ``getVariable``, ``getConstraints``,
``getConstraint``, ``getNumVariables``, ``getNumConstraints``,
``getNumNonZeros``, ``isMip``, ``isSolved``, ``getStatus``,
``getObjectiveValue``, and ``getSolveTime``.

Low-level data model
--------------------

``DataModel`` is useful when the problem is already available as arrays or
when direct access to native CSR and quadratic data is required.

Create a model with one of the following factories:

.. code-block:: java

   DataModel.createProblem(
       numConstraints, numVariables, objectiveSense, objectiveOffset,
       objectiveCoefficients, constraintMatrix, constraintSense, rhs,
       variableLowerBounds, variableUpperBounds, variableTypes);

   DataModel.createRangedProblem(
       numConstraints, numVariables, objectiveSense, objectiveOffset,
       objectiveCoefficients, constraintMatrix, constraintLowerBounds,
       constraintUpperBounds, variableLowerBounds, variableUpperBounds,
       variableTypes);

``CsrMatrix`` stores ``rowOffsets``, ``columnIndices``, and ``values``. The
arrays are available through ``getRowOffsets``, ``getColumnIndices``, and
``getValues``.

The mutable ``DataModel`` setters cover:

* objective sense, coefficients, offset, and scaling factor;
* linear constraint CSR arrays, row types, RHS, and ranged bounds;
* variable bounds, types, names, and row names;
* objective and problem names;
* initial primal and dual solutions; and
* quadratic objective matrices and quadratic constraints.

The corresponding getters include ``getConstraintMatrix``,
``getConstraintMatrixValues``, ``getConstraintMatrixIndices``,
``getConstraintMatrixOffsets``, ``getConstraintRhs``,
``getConstraintLowerBounds``, ``getConstraintUpperBounds``,
``getQuadraticObjectiveValues``, ``getQuadraticObjectiveIndices``,
``getQuadraticObjectiveOffsets``, ``getQuadraticConstraints``,
``getVariableNames``, ``getRowNames``, ``getObjectiveName``,
``getProblemName``, ``getProblemCategory``, and ``toDict``.

Use ``clearQuadraticConstraints`` to remove all quadratic constraints from a
mutable data model. ``DataModel`` implements ``AutoCloseable``.

Variables, expressions, and constraints
----------------------------------------

``Variable`` stores the model index, bounds, objective coefficient, type,
name, solved value, reduced cost, and optional MIP start. Its mutable methods
return the variable so calls can be chained:

.. code-block:: java

   Variable x = problem.addVariable(
       0.0, Double.POSITIVE_INFINITY, 1.0,
       VariableType.CONTINUOUS, "x");
   x.setUpperBound(100.0).setObjectiveCoefficient(2.0);

``LinearExpression`` supports ``of``, ``ofConstant``, ``plus``, ``minus``,
``times``, ``dividedBy``, ``constant``, and the comparison methods ``le``,
``ge``, and ``eq``. Comparisons return a ``Constraint``.

``QuadraticExpression`` supports quadratic terms through
``QuadraticExpression.of(first, second, coefficient)`` and the same fluent
arithmetic pattern. It can also contain linear and constant terms. Its
``le`` and ``ge`` methods return quadratic constraints; ``eq`` throws because
equality quadratic constraints are not supported.

The enums used in model construction are:

* ``ObjectiveSense.MINIMIZE`` and ``ObjectiveSense.MAXIMIZE``;
* ``ConstraintSense.LE``, ``ConstraintSense.GE``, and ``ConstraintSense.EQ``;
* ``VariableType.CONTINUOUS``, ``VariableType.INTEGER``, and
  ``VariableType.SEMI_CONTINUOUS``; and
* ``ProblemCategory`` for the native problem classification.

``Constraint`` provides ``getSense``, ``getRHS``, ``getCoefficient``,
``getLinearExpression``, ``getQuadraticExpression``, ``isQuadratic``,
``computeSlack``, ``getSlack``, and ``getDualValue``.

Solver settings
---------------

``SolverSettings`` owns native solver configuration and implements
``AutoCloseable``. Parameters can be set with the overloaded
``setParameter`` methods for ``String``, ``int``, ``double``, and ``boolean``
values. Use ``getParameter`` or ``getParameterAsString`` for the native string
representation, and ``getTypedParameter`` when a Java ``Boolean``,
``Integer``, ``Double``, or ``String`` value is preferred.

The settings API also includes:

* ``getSolverParameterNames`` and the static setting accessors;
* ``setMethod`` and ``setPdlpSolverMode``;
* ``setOptimalityTolerance``;
* primal and dual initial solutions;
* ``dumpParametersToFile`` and ``loadParametersFromFile``;
* ``toDict``;
* ``setPdlpWarmStartData``; and
* MIP callback registration through ``setMipCallback``.

``SolverMethod`` includes ``PDLP``, ``DUAL_SIMPLEX``, ``BARRIER``,
``CONCURRENT``, and ``UNSET``. ``PDLPSolverMode`` exposes the supported PDLP
solver modes.

Solutions and statistics
------------------------

``Solution`` implements ``AutoCloseable`` and exposes:

* ``getPrimalSolution``, ``getDualSolution``, and ``getReducedCost``;
* ``getPrimalObjective`` and ``getDualObjective``;
* ``getTerminationStatus`` and ``getTerminationReason``;
* ``getErrorStatus`` and ``getErrorMessage``;
* ``getSolveTime`` and ``getProblemCategory``; and
* ``getVars`` when variable names are available.

LP solutions additionally expose ``getLpStats`` and PDLP warm-start data.
``LPStats`` contains primal residual, dual residual, gap, iteration count, and
the ``SolverMethod`` used. MIP-only solution fields are documented in
:doc:`../mip/mip-api`.

MPS, batching, and errors
-------------------------

``DataModel.read``, ``DataModel.parseMps``, ``Problem.read``, and
``Problem.readMPS`` support MPS/QPS parsing, including a fixed-format boolean
overload. ``writeMPS`` writes a model for round trips or use by another cuOpt
interface.

``BatchSolve.solve(List<DataModel>, SolverSettings)`` is a sequential Java
compatibility entry point. It returns ``BatchSolveResult``, containing the
solutions and elapsed solve time.

Native failures are reported as ``CuOptException`` with a cuOpt status code
available through ``getStatusCode``. Accessing an LP-only field on a MIP
solution, or a MIP-only field on an LP solution, raises
``IllegalStateException``.
