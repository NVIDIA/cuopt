cuOpt Convex Optimization C API Reference
=========================================

This section contains the cuOpt convex optimization C API reference. For MIP-specific functions and callbacks, see :doc:`../mip/mip-c-api`.

Integer and Floating-Point Types
---------------------------------

cuOpt may be built with 32 or 64 bit integer and floating-point types. The C API uses a `typedef` for floating point and integer types to abstract the size of these types.

.. doxygentypedef:: cuopt_int_t
.. doxygentypedef:: cuopt_float_t

You may use the following functions to determine the number of bytes used to represent these types in your build

.. doxygenfunction:: cuOptGetIntSize
.. doxygenfunction:: cuOptGetFloatSize

Version Information
-------------------

You may use the following function to get the version of the cuOpt library

.. doxygenfunction:: cuOptGetVersion

Status Codes
------------

Every function in the C API returns a status code that indicates success or failure. The following status codes are defined

.. doxygendefine:: CUOPT_SUCCESS
.. doxygendefine:: CUOPT_INVALID_ARGUMENT
.. doxygendefine:: CUOPT_MPS_FILE_ERROR
.. doxygendefine:: CUOPT_MPS_PARSE_ERROR
.. doxygendefine:: CUOPT_VALIDATION_ERROR
.. doxygendefine:: CUOPT_OUT_OF_MEMORY
.. doxygendefine:: CUOPT_RUNTIME_ERROR

Optimization Problem
--------------------

An optimization problem is represented via a `cuOptOptimizationProblem`

.. doxygentypedef:: cuOptOptimizationProblem

Optimization problems can be created, loaded, or written via the following functions:

.. doxygenfunction:: cuOptReadProblem
.. doxygenfunction:: cuOptWriteProblem
.. doxygenfunction:: cuOptCreateProblem
.. doxygenfunction:: cuOptCreateRangedProblem

.. note::
   ``cuOptCreateQuadraticProblem`` and ``cuOptCreateQuadraticRangedProblem`` are deprecated.
   Prefer ``cuOptCreateProblem`` or ``cuOptCreateRangedProblem`` followed by
   ``cuOptSetQuadraticObjective``.

For problems with quadratic objectives, first create a problem, and then use

.. doxygenfunction:: cuOptSetQuadraticObjective

For problems with quadratic constraints, first create a problem, and then use

.. doxygenfunction:: cuOptAddQuadraticConstraint

.. note::
   Support for quadratic constraints is currently in **beta**. ``cuOptAddQuadraticConstraint``
   supports three types of quadratic constraints:
   - **Convex quadratic constraints** of the form ``x^T Q x + d^T x <= alpha`` where ``H=(Q+Q^T)/2`` is a symmetric positive semidefinite matrix. `Q` need not be symmetric.
   - **Second-order cone constraints** of the form ``sum_{i=1}^n x_i^2 <= x_0``, ``x_0 >= 0``
   - **Rotated second-order cone constraints** of the form ``sum_{i=2}^n x_i^2  - 0. 5 * x_0 * x_1 - 0. 5 * x_1 * x_0 <= 0``, ``x_0 >= 0``, ``x_1 >= 0``

   For the rotated second-order cone constraints, cuOpt expects the quadratic matrix to be symmetric.
   Only ``CUOPT_LESS_THAN`` and ``CUOPT_GREATER_THAN`` sense is supported; equality constraints are not supported.

A optimization problem must be destroyed with the following function

.. doxygenfunction:: cuOptDestroyProblem

Certain constants are needed to define an optimization problem. These constants are described below.

Objective Sense Constants
-------------------------

These constants are used to define the objective sense in the :c:func:`cuOptCreateProblem` and :c:func:`cuOptCreateRangedProblem` functions.

.. doxygendefine:: CUOPT_MINIMIZE
.. doxygendefine:: CUOPT_MAXIMIZE

Constraint Sense Constants
--------------------------

These constants are used to define the constraint sense in the :c:func:`cuOptCreateProblem` and :c:func:`cuOptCreateRangedProblem` functions.

.. doxygendefine:: CUOPT_LESS_THAN
.. doxygendefine:: CUOPT_GREATER_THAN
.. doxygendefine:: CUOPT_EQUAL

Variable Type Constants
-----------------------

These constants are used to define the the variable type in the :c:func:`cuOptCreateProblem` and :c:func:`cuOptCreateRangedProblem` functions.

.. doxygendefine:: CUOPT_CONTINUOUS
.. doxygendefine:: CUOPT_INTEGER
.. doxygendefine:: CUOPT_SEMI_CONTINUOUS

Infinity Constant
-----------------

This constant may be used to represent infinity in the :c:func:`cuOptCreateProblem` and :c:func:`cuOptCreateRangedProblem` functions.

.. doxygendefine:: CUOPT_INFINITY

File Format Constants
---------------------

These constants are used to specify the output file format in :c:func:`cuOptWriteProblem`.

.. doxygendefine:: CUOPT_FILE_FORMAT_MPS

Querying an Optimization Problem
--------------------------------

The following functions may be used to get information about an `cuOptimizationProblem`

.. doxygenfunction:: cuOptGetNumConstraints
.. doxygenfunction:: cuOptGetNumVariables
.. doxygenfunction:: cuOptGetObjectiveSense
.. doxygenfunction:: cuOptGetObjectiveOffset
.. doxygenfunction:: cuOptGetObjectiveCoefficients
.. doxygenfunction:: cuOptGetNumNonZeros
.. doxygenfunction:: cuOptGetConstraintMatrix
.. doxygenfunction:: cuOptGetConstraintSense
.. doxygenfunction:: cuOptGetConstraintRightHandSide
.. doxygenfunction:: cuOptGetConstraintLowerBounds
.. doxygenfunction:: cuOptGetConstraintUpperBounds
.. doxygenfunction:: cuOptGetVariableLowerBounds
.. doxygenfunction:: cuOptGetVariableUpperBounds
.. doxygenfunction:: cuOptGetVariableTypes
.. doxygenfunction:: cuOptIsMIP


Solver Settings
---------------

Settings are used to configure the LP/MIP solvers. All settings are stored in a `cuOptSolverSettings` object.


.. doxygentypedef:: cuOptSolverSettings

A `cuOptSolverSettings` object is created with `cuOptCreateSolverSettings`

.. doxygenfunction:: cuOptCreateSolverSettings

When you are done with a solve you should destroy a `cuOptSolverSettings` object with

.. doxygenfunction:: cuOptDestroySolverSettings


Setting Parameters
------------------
The following functions are used to set and get parameters. You can find more details on the available parameters in the :doc:`Convex Optimization Settings <../../convex-settings>` section.

.. doxygenfunction:: cuOptSetParameter
.. doxygenfunction:: cuOptGetParameter
.. doxygenfunction:: cuOptSetIntegerParameter
.. doxygenfunction:: cuOptGetIntegerParameter
.. doxygenfunction:: cuOptSetFloatParameter
.. doxygenfunction:: cuOptGetFloatParameter

.. _parameter-constants:

Parameter Constants
-------------------

These constants are used as parameter names in the :c:func:`cuOptSetParameter`, :c:func:`cuOptGetParameter`, and similar functions. For more details on the available parameters, see the :doc:`Convex Optimization Settings <../../convex-settings>` and :doc:`MIP Settings <../../mip-settings>` sections.

.. Convex optimization (LP/QP/QCQP/SOCP) parameter string constants
.. doxygendefine:: CUOPT_ABSOLUTE_DUAL_TOLERANCE
.. doxygendefine:: CUOPT_RELATIVE_DUAL_TOLERANCE
.. doxygendefine:: CUOPT_ABSOLUTE_PRIMAL_TOLERANCE
.. doxygendefine:: CUOPT_RELATIVE_PRIMAL_TOLERANCE
.. doxygendefine:: CUOPT_ABSOLUTE_GAP_TOLERANCE
.. doxygendefine:: CUOPT_RELATIVE_GAP_TOLERANCE
.. doxygendefine:: CUOPT_INFEASIBILITY_DETECTION
.. doxygendefine:: CUOPT_STRICT_INFEASIBILITY
.. doxygendefine:: CUOPT_PRIMAL_INFEASIBLE_TOLERANCE
.. doxygendefine:: CUOPT_DUAL_INFEASIBLE_TOLERANCE
.. doxygendefine:: CUOPT_ITERATION_LIMIT
.. doxygendefine:: CUOPT_TIME_LIMIT
.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE
.. doxygendefine:: CUOPT_METHOD
.. doxygendefine:: CUOPT_PER_CONSTRAINT_RESIDUAL
.. doxygendefine:: CUOPT_SAVE_BEST_PRIMAL_SO_FAR
.. doxygendefine:: CUOPT_FIRST_PRIMAL_FEASIBLE
.. doxygendefine:: CUOPT_LOG_FILE
.. doxygendefine:: CUOPT_PRESOLVE
.. doxygendefine:: CUOPT_LOG_TO_CONSOLE
.. doxygendefine:: CUOPT_CROSSOVER
.. doxygendefine:: CUOPT_FOLDING
.. doxygendefine:: CUOPT_AUGMENTED
.. doxygendefine:: CUOPT_DUALIZE
.. doxygendefine:: CUOPT_ORDERING
.. doxygendefine:: CUOPT_ELIMINATE_DENSE_COLUMNS
.. doxygendefine:: CUOPT_CUDSS_DETERMINISTIC
.. doxygendefine:: CUOPT_BARRIER_DUAL_INITIAL_POINT
.. doxygendefine:: CUOPT_BARRIER_ITERATIVE_REFINEMENT
.. doxygendefine:: CUOPT_BARRIER_STEP_SCALE
.. doxygendefine:: CUOPT_DUAL_POSTSOLVE
.. doxygendefine:: CUOPT_SOLUTION_FILE
.. doxygendefine:: CUOPT_NUM_CPU_THREADS
.. doxygendefine:: CUOPT_NUM_GPUS
.. doxygendefine:: CUOPT_USER_PROBLEM_FILE
.. doxygendefine:: CUOPT_PDLP_PRECISION

.. _pdlp-solver-mode-constants:

PDLP Solver Mode Constants
--------------------------

These constants are used to configure `CUOPT_PDLP_SOLVER_MODE` via :c:func:`cuOptSetIntegerParameter`.

.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_STABLE1
.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_STABLE2
.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_STABLE3
.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_METHODICAL1
.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_FAST1

.. _pdlp-precision-constants:

PDLP Precision Constants
------------------------

These constants are used to configure `CUOPT_PDLP_PRECISION` via :c:func:`cuOptSetIntegerParameter`.

.. doxygendefine:: CUOPT_PDLP_DEFAULT_PRECISION
.. doxygendefine:: CUOPT_PDLP_SINGLE_PRECISION
.. doxygendefine:: CUOPT_PDLP_DOUBLE_PRECISION
.. doxygendefine:: CUOPT_PDLP_MIXED_PRECISION

.. _method-constants:

Method Constants
----------------

These constants are used to configure `CUOPT_METHOD` via :c:func:`cuOptSetIntegerParameter`.

.. doxygendefine:: CUOPT_METHOD_CONCURRENT
.. doxygendefine:: CUOPT_METHOD_PDLP
.. doxygendefine:: CUOPT_METHOD_DUAL_SIMPLEX
.. doxygendefine:: CUOPT_METHOD_BARRIER
.. doxygendefine:: CUOPT_METHOD_UNSET

.. _barrier-iterative-refinement-constants:

Barrier Iterative Refinement Constants
---------------------------------------

These constants are used to configure `CUOPT_BARRIER_ITERATIVE_REFINEMENT` via :c:func:`cuOptSetIntegerParameter`.

.. doxygendefine:: CUOPT_BARRIER_ITERATIVE_REFINEMENT_OFF
.. doxygendefine:: CUOPT_BARRIER_ITERATIVE_REFINEMENT_ON


Warm Start
----------

For LP problems solved with PDLP, primal and dual warm start vectors may be provided:

.. doxygenfunction:: cuOptSetInitialPrimalSolution
.. doxygenfunction:: cuOptSetInitialDualSolution

.. note::
   For MIP warm start (MIP starts), see :doc:`../mip/mip-c-api`.


Solving an LP or MIP
--------------------

LP and MIP solves are performed by calling the `cuOptSolve` function

.. doxygenfunction:: cuOptSolve


Solution
--------

The output of a solve is a `cuOptSolution` object.

.. doxygentypedef:: cuOptSolution

The following functions may be used to access information from a `cuOptSolution`

.. doxygenfunction:: cuOptGetTerminationStatus
.. doxygenfunction:: cuOptGetErrorStatus
.. doxygenfunction:: cuOptGetErrorString
.. doxygenfunction:: cuOptGetPrimalSolution
.. doxygenfunction:: cuOptGetObjectiveValue
.. doxygenfunction:: cuOptGetSolveTime
.. doxygenfunction:: cuOptGetMIPGap
.. doxygenfunction:: cuOptGetSolutionBound
.. doxygenfunction:: cuOptGetDualSolution
.. doxygenfunction:: cuOptGetDualObjectiveValue
.. doxygenfunction:: cuOptGetReducedCosts

When you are finished with a `cuOptSolution` object you should destory it with

.. doxygenfunction:: cuOptDestroySolution

Termination Status Constants
----------------------------

These constants define the termination status received from the :c:func:`cuOptGetTerminationStatus` function.

.. LP/MIP termination status constants
.. doxygendefine:: CUOPT_TERMINATION_STATUS_NO_TERMINATION
.. doxygendefine:: CUOPT_TERMINATION_STATUS_OPTIMAL
.. doxygendefine:: CUOPT_TERMINATION_STATUS_INFEASIBLE
.. doxygendefine:: CUOPT_TERMINATION_STATUS_UNBOUNDED
.. doxygendefine:: CUOPT_TERMINATION_STATUS_ITERATION_LIMIT
.. doxygendefine:: CUOPT_TERMINATION_STATUS_TIME_LIMIT
.. doxygendefine:: CUOPT_TERMINATION_STATUS_NUMERICAL_ERROR
.. doxygendefine:: CUOPT_TERMINATION_STATUS_PRIMAL_FEASIBLE
.. doxygendefine:: CUOPT_TERMINATION_STATUS_FEASIBLE_FOUND
.. doxygendefine:: CUOPT_TERMINATION_STATUS_CONCURRENT_LIMIT
.. doxygendefine:: CUOPT_TERMINATION_STATUS_WORK_LIMIT
.. doxygendefine:: CUOPT_TERMINATION_STATUS_UNBOUNDED_OR_INFEASIBLE
