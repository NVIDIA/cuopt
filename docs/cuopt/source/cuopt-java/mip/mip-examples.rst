============
MIP Examples
============

These examples show mixed-integer modeling, MIP starts, semi-continuous
variables, and incumbent callbacks in Java.

Simple MIP
----------

:download:`SimpleMip.java <examples/SimpleMip.java>`

.. literalinclude:: examples/SimpleMip.java
   :language: java
   :linenos:

Example Response:

.. code-block:: text

   Status: OPTIMAL
   x = 0.0
   y = 8.0
   Objective = 40.0
   MIP gap = 0.0
   Bound = 40.0

The MIP solver can return a feasible solution before proving optimality. Use
the termination status, MIP gap, and solution bound together when interpreting
the result.

Semi-Continuous Variables
-------------------------

``SEMI_CONTINUOUS`` variables are zero or lie within their declared bounds.

:download:`SemiContinuous.java <examples/SemiContinuous.java>`

.. literalinclude:: examples/SemiContinuous.java
   :language: java
   :linenos:

MIP Starts
----------

Set starts on variables when using the high-level ``Problem`` API:

:download:`MipStarts.java <examples/MipStarts.java>`

.. literalinclude:: examples/MipStarts.java
   :language: java
   :start-after: // start-per-variable
   :end-before: // end-per-variable
   :dedent:

Setting a start per variable avoids handling the ordering at all, and is the
form to prefer.

``SolverSettings.addMIPStart`` takes a complete array instead, indexed by
``Variable.getIndex()``. It is the way to supply more than one starting point,
since it can be called repeatedly while each ``Variable`` holds a single value.
Build it from ``getVariables`` so the ordering comes from the problem rather
than from you:

.. literalinclude:: examples/MipStarts.java
   :language: java
   :start-after: // start-array
   :end-before: // end-array
   :dedent:

MIP starts are currently unsupported with presolve on.

Incumbent Callback
------------------

Register an incumbent callback before solving:

:download:`IncumbentCallback.java <examples/IncumbentCallback.java>`

.. literalinclude:: examples/IncumbentCallback.java
   :language: java
   :start-after: // start-basic-callback
   :end-before: // end-basic-callback
   :dedent:

The callback receives a defensive copy of the incumbent vector, the incumbent
objective, the current solution bound, and the user data object.

The vector is in variable-index order. To read specific variables out of it
without depending on that order, pass them to ``Problem.fromIncumbent``, which
returns their values in the order you ask for:

.. literalinclude:: examples/IncumbentCallback.java
   :language: java
   :start-after: // start-from-incumbent
   :end-before: // end-from-incumbent
   :dedent:

LP Relaxation
-------------

Relax the integer variables before solving:

:download:`LpRelaxation.java <examples/LpRelaxation.java>`

.. literalinclude:: examples/LpRelaxation.java
   :language: java
   :start-after: // start-relax
   :end-before: // end-relax
   :dedent:
