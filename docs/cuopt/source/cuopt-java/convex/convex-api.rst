===================================
Convex Optimization API Reference
===================================

The Java LP/MIP/QP bindings are in the package
``com.nvidia.cuopt.mathematicalprogramming``. The public API is documented below by
role. Method names are Java names and therefore use fluent methods instead of
Python operator overloads.

High-Level Problem
------------------

``Problem`` is the recommended entry point for problems built in Java.

.. list-table:: ``Problem``
   :header-rows: 1
   :widths: 28 72

   * - API
     - Description
   * - ``new Problem()`` / ``new Problem(String name)``
     - Create an empty problem, optionally with a name.
   * - ``addVariable(...)``
     - Add a variable with lower/upper bounds, objective coefficient, variable type, and name.
   * - ``addConstraint(Constraint, String name)``
     - Add a linear or quadratic constraint.
   * - ``setObjective(LinearExpression, ObjectiveSense)``
     - Set a linear objective.
   * - ``setObjective(QuadraticExpression, ObjectiveSense)``
     - Set a quadratic objective with optional linear and constant terms.
   * - ``solve()`` / ``solve(SolverSettings)``
     - Solve the problem and return a ``Solution``.
   * - ``getConstraintMatrix()`` / ``getQuadraticObjectiveMatrix()``
     - Inspect the linear constraint matrix, or the quadratic objective matrix Q, in CSR form.
   * - ``read(String)`` / ``write(String)``
     - Load or write a problem. The format follows the file extension; a fixed-format MPS
       overload of ``read`` accepts a boolean flag.
   * - ``update()`` / ``updateConstraint(...)`` / ``updateObjective(...)``
     - Update problem state and reset solved values where appropriate.
   * - ``relax()``
     - Return a copy with variables converted to continuous type.

``Problem`` also exposes ``getVariables``, ``getVariable``, ``getConstraints``,
``getConstraint``, ``getNumVariables``, ``getNumConstraints``,
``getNumNonZeros``, ``isMIP``, ``isSolved``, ``getStatus``,
``getObjective``, ``getObjectiveValue``, and ``getSolveTime``. ``getObjective``
returns the common ``ObjectiveExpression`` type; ``isQuadratic`` distinguishes
the concrete linear and quadratic forms without an ``Object`` downcast.

``CSRMatrix`` takes ``values``, ``columnIndices``, and ``rowOffsets`` in the
same order used by cuOpt CSR arrays. The arrays are available through
``getValues``, ``getColumnIndices``, and ``getRowOffsets``.

Variables, Expressions, and Constraints
----------------------------------------

``Variable`` stores the problem index, bounds, objective coefficient, type,
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
``le`` and ``ge`` methods return quadratic constraints. It does not expose an
``eq`` method because equality quadratic constraints are not supported.

Both expression classes implement ``ObjectiveExpression``, which exposes the
linear portion, constant, current value, and ``isQuadratic``.

The enums used in problem construction are:

* ``ObjectiveSense.MINIMIZE`` and ``ObjectiveSense.MAXIMIZE``;
* ``ConstraintSense.LE``, ``ConstraintSense.GE``, and ``ConstraintSense.EQ``;
* ``VariableType.CONTINUOUS``, ``VariableType.INTEGER``, and
  ``VariableType.SEMI_CONTINUOUS``; and
* ``ProblemCategory`` for the native problem classification.

``Constraint`` provides ``getSense``, ``getRHS``, ``getCoefficient``,
``getLinearExpression``, ``getQuadraticExpression``, ``isQuadratic``,
``computeSlack``, ``getSlack``, and ``getDualValue``.

Solver Settings
---------------

``SolverSettings`` owns native solver configuration and implements
``AutoCloseable``. Settings can be set with the overloaded
``setSetting`` methods for ``String``, ``int``, ``double``, and ``boolean``
values. Use ``getSetting`` or ``getSettingAsString`` for the native string
representation. The ``getSetting(name, type)`` overload provides a typed
``Boolean``, ``Integer``, ``Double``, or ``String`` result, for example
``getSetting(CuOptConstants.CUOPT_TIME_LIMIT, Double.class)``.

The settings API also includes:

* the static setting accessors;
* ``setMethod`` and ``setPDLPSolverMode``;
* ``setOptimalityTolerance``;
* MIP callback registration through ``setMIPCallback``.

``SolverMethod`` includes ``PDLP``, ``DUAL_SIMPLEX``, ``BARRIER``,
``CONCURRENT``, and ``UNSET``. ``PDLPSolverMode`` exposes the supported PDLP
solver modes.

Solutions and Statistics
------------------------

``Solution`` implements ``AutoCloseable`` and exposes:

* ``getPrimalSolution``, ``getDualSolution``, and ``getReducedCost``;
* ``getPrimalObjective`` and ``getDualObjective``;
* ``getTerminationStatus`` and ``getTerminationReason``;
* ``getErrorStatus`` and ``getErrorMessage``;
* ``getSolveTime`` and ``getProblemCategory``; and
* ``getVars`` when variable names are available.

LP solutions additionally expose ``getLPStats``. ``LPStats`` contains primal
residual, dual residual, gap, iteration count, and the ``SolverMethod`` used.
MIP-only solution fields are documented in :doc:`../mip/mip-api`.

MPS and Errors
--------------

``Problem.read`` loads a problem, choosing the parser from the file extension,
and takes a boolean overload to force fixed-format MPS. ``Problem.write``
writes a problem for round trips or use by another cuOpt interface.

Native failures are reported as ``CuOptException`` with a cuOpt status code
available through ``getStatusCode``. Accessing an LP-only field on a MIP
solution, or a MIP-only field on an LP solution, raises
``IllegalStateException``.
