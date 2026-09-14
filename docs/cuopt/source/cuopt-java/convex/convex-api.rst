===================================
Convex Optimization API Reference
===================================

The Java LP/QP/QCQP/SOCP bindings are in the package
``com.nvidia.cuopt.mathematicaloptimization``. The public API is documented below by
role. Expressions and comparisons are formed through fluent methods. MIP is
documented separately in :doc:`../mip/index`.

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
   * - ``fromIncumbent(double[], Variable...)``
     - Static. Read the given variables out of an array in variable-index order, returning
       their values in the order asked for.
   * - ``read(String)`` / ``write(String)``
     - Load a problem, choosing the parser from the file extension, or write one as MPS.
       A fixed-format MPS overload of ``read`` accepts a boolean flag.

``Problem`` also exposes ``getVariables``, ``getVariable``, ``getConstraints``,
``getConstraint``, ``getNumVariables``, ``getNumConstraints``,
``getNumNonZeros``, ``isMIP``, ``getStatus``, ``getObjective``,
``getObjectiveValue``, and ``getSolveTime``. ``getObjective`` returns the linear
part of the objective; the quadratic part, when there is one, is available as a
matrix from ``getQuadraticObjectiveMatrix``.

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

The enums used in problem construction are:

* ``ObjectiveSense.MINIMIZE`` and ``ObjectiveSense.MAXIMIZE``;
* ``ConstraintSense.LE``, ``ConstraintSense.GE``, and ``ConstraintSense.EQ``;
* ``VariableType.CONTINUOUS``, ``VariableType.INTEGER``, and
  ``VariableType.SEMI_CONTINUOUS``.

``Constraint`` provides ``getSense``, ``getRHS``, ``getCoefficient``,
``getLinearExpression``, ``getQuadraticExpression``, ``isQuadratic``,
``getSlack``, and ``getDualValue``.

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
* ``setOptimalityTolerance``.

``SolverMethod`` includes ``PDLP``, ``DUAL_SIMPLEX``, ``BARRIER`` and
``CONCURRENT``. ``PDLPSolverMode`` exposes the supported PDLP solver modes.

Solutions and Statistics
------------------------

``Solution`` implements ``AutoCloseable`` and exposes:

* ``getPrimalObjective`` and ``getDualObjective``;
* ``getTerminationStatus``;
* ``getErrorStatus`` and ``getErrorMessage``.

Solution values are read from the model rather than as bulk arrays:
``Variable.getValue`` and ``Variable.getReducedCost`` for variables, and
``Constraint.getDualValue`` and ``Constraint.getSlack`` for constraints. They
are populated on the ``Problem`` after each solve.

Solver statistics are read as scalar solution attributes through
``getIntAttribute`` and ``getFloatAttribute``, selected by a
``CuOptConstants.CUOPT_SOLUTION_ATTR_*`` value, so a statistic added later
becomes a new constant rather than a new method.

.. code-block:: java

   double gap = solution.getFloatAttribute(
       CuOptConstants.CUOPT_SOLUTION_ATTR_LP_GAP);
   int iterations = solution.getIntAttribute(
       CuOptConstants.CUOPT_SOLUTION_ATTR_LP_NUM_ITERATIONS);

This depends on the class of problem that produced the solution. A selector
that does not apply, or that does not have the requested value type, raises
``CuOptException``.

Reading and Writing Problems
----------------------------

``Problem.read`` loads a problem, choosing the parser from the file extension:
``.mps``, ``.qps`` and ``.lp`` are recognised, along with their ``.gz``,
``.bz2`` and ``.lz4`` variants. A Boolean overload forces fixed-format MPS.

``Problem.write`` writes MPS, which is the only format the engine can write, so
the path must end in ``.mps`` or ``.qps``.

Errors
------

Native failures are reported as ``CuOptException`` with a cuOpt status code
available through ``getStatusCode``. Reading a field that does not apply to
the class of problem that produced the solution raises
``IllegalStateException``.
