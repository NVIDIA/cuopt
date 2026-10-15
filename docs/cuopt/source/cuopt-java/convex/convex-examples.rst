============================
Convex Optimization Examples
============================

These examples show the Java modeling patterns for LP and QP problems. They
assume the Java module has been compiled as described in
:doc:`../quick-start` and that the application can load ``libcuopt_jni``.

Simple Linear Programming
--------------------------

The high-level API uses fluent expressions and explicit comparison methods.

:download:`SimpleLp.java <examples/SimpleLp.java>`

.. literalinclude:: examples/SimpleLp.java
   :language: java
   :linenos:

Example Response:

.. code-block:: text

   Status: OPTIMAL
   x = 0.0
   y = 10.0
   Objective = 10.0

``Problem.solve`` populates the ``Variable`` and ``Constraint`` objects after
the solve. The solution object remains available for detailed native results
and statistics.

Simple Quadratic Programming
-----------------------------

Quadratic objectives combine quadratic, linear, and constant terms:

:download:`SimpleQp.java <examples/SimpleQp.java>`

.. literalinclude:: examples/SimpleQp.java
   :language: java
   :linenos:

Example Response:

.. code-block:: text

   x = 0.5
   y = 0.5
   Objective = -0.5

For QP solutions, ``getDualObjective`` is available when the solver returns it,
and variable and constraint values are read from the model through
``Variable.getValue``, ``Variable.getReducedCost``, and
``Constraint.getDualValue``.

Quadratic Constraints
---------------------

Quadratic constraints can be added directly to a ``Problem``. As of this
writing, ``cuOptCreateProblem`` requires at least one linear constraint row,
so a purely quadratically-constrained model needs a (possibly non-binding)
linear constraint too:

:download:`QuadraticConstraint.java <examples/QuadraticConstraint.java>`

.. literalinclude:: examples/QuadraticConstraint.java
   :language: java
   :linenos:

Only ``LE`` and ``GE`` quadratic constraints are supported;
``QuadraticExpression`` does not expose an ``eq`` method.

Reading and Writing MPS/QPS
---------------------------

``Problem`` exposes both extension-dispatch and direct MPS entry points:

:download:`MpsRoundtrip.java <examples/MpsRoundtrip.java>` and
:download:`sample.mps <examples/sample.mps>`

.. literalinclude:: examples/MpsRoundtrip.java
   :language: java
   :linenos:

Example Response:

.. code-block:: text

   Variables: 2

Fixed-format parsing is also available:

.. code-block:: java

   try (Problem fixed = Problem.read("fixed-format.mps", true)) {
     // Use fixed-format parsing explicitly.
   }

Parsing failures are reported as ``CuOptException`` with the cuOpt status code
available from ``getStatusCode``.
