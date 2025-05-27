cuOpt LP/MILP C API Reference
========================================

This section contains the cuOpt LP/MILP C API reference.

Integer and Floating-Point Types
---------------------------------

cuOpt may be built with 32 or 64 bit integer and floating-point types. The C API uses a `typedef` for floating point and integer types to abstract the size of these types.

.. doxygentypedef:: cuopt_int_t
.. doxygentypedef:: cuopt_float_t

You may use the following functions to determine the number of bytes used to represent these types in your build

.. doxygenfunction:: cuOptGetIntSize
.. doxygenfunction:: cuOptGetFloatSize

Status Codes
------------

Every function in the C API returns a status code that indicates success or failure. The following status codes are defined

.. doxygendefine:: CUOPT_SUCCESS
.. doxygendefine:: CUOPT_INVALID_ARGUMENT
.. doxygendefine:: CUOPT_MPS_FILE_ERROR
.. doxygendefine:: CUOPT_MPS_PARSE_ERROR

Optimization Problem
--------------------

An optimization problem is represented via a `cuOptOptimizationProblem`

.. doxygentypedef:: cuOptOptimizationProblem

Optimization problems can be created via three different functions

.. doxygenfunction:: cuOptReadProblem
.. doxygenfunction:: cuOptCreateProblem
.. doxygenfunction:: cuOptCreateRangedProblem

A optimization problem must be destroyed with the following function

.. doxygenfunction:: cuOptDestroyProblem

Certain constants are needed to define an optimization problem. These constants are described below.

Objective Sense Constants
-------------------------

These constants are used to define the objective sense in the `cuOptCreateProblem <lp-milp-c-api.html#c.cuOptCreateProblem>`_ and `cuOptCreateRangedProblem <lp-milp-c-api.html#c.cuOptCreateRangedProblem>`_ functions.

.. doxygendefine:: CUOPT_MINIMIZE
.. doxygendefine:: CUOPT_MAXIMIZE

Constraint Sense Constants
--------------------------

These constants are used to define the constraint sense in the `cuOptCreateProblem <lp-milp-c-api.html#c.cuOptCreateProblem>`_ and `cuOptCreateRangedProblem <lp-milp-c-api.html#c.cuOptCreateRangedProblem>`_ functions.

.. doxygendefine:: CUOPT_LESS_THAN
.. doxygendefine:: CUOPT_GREATER_THAN
.. doxygendefine:: CUOPT_EQUAL

Variable Type Constants
-----------------------

These constants are used to define the the variable type in the `cuOptCreateProblem <lp-milp-c-api.html#c.cuOptCreateProblem>`_ and `cuOptCreateRangedProblem <lp-milp-c-api.html#c.cuOptCreateRangedProblem>`_ functions.

.. doxygendefine:: CUOPT_CONTINUOUS
.. doxygendefine:: CUOPT_INTEGER

Infinity Constant
-----------------

This constant may be used to represent infinity in the `cuOptCreateProblem <lp-milp-c-api.html#c.cuOptCreateProblem>`_ and `cuOptCreateRangedProblem <lp-milp-c-api.html#c.cuOptCreateRangedProblem>`_ functions.

.. doxygendefine:: CUOPT_INFINITY

Querying an optimization problem
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
The following functions are used to set and get parameters. You can find more details on the available parameters in the `LP/MILP settings <../../lp-milp-settings.html>`_ section.

.. doxygenfunction:: cuOptSetParameter
.. doxygenfunction:: cuOptGetParameter
.. doxygenfunction:: cuOptSetIntegerParameter
.. doxygenfunction:: cuOptGetIntegerParameter
.. doxygenfunction:: cuOptSetFloatParameter
.. doxygenfunction:: cuOptGetFloatParameter


Parameter Constants
------------------- 

These constants are used as the parameter name in the `cuOptSetParameter <lp-milp-c-api.html#c.cuOptSetParameter>`_ , `cuOptGetParameter <lp-milp-c-api.html#c.cuOptGetParameter>`_ and similar functions. More details on the parameters can be found in the `LP/MILP settings <../../lp-milp-settings.html>`_ section.

.. LP/MIP parameter string constants
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
.. doxygendefine:: CUOPT_MIP_ABSOLUTE_TOLERANCE
.. doxygendefine:: CUOPT_MIP_RELATIVE_TOLERANCE
.. doxygendefine:: CUOPT_MIP_INTEGRALITY_TOLERANCE
.. doxygendefine:: CUOPT_MIP_SCALING
.. doxygendefine:: CUOPT_MIP_HEURISTICS_ONLY
.. doxygendefine:: CUOPT_NUM_CPU_THREADS

PDLP Solver Mode Constants
--------------------------

These constants are used to configure `CUOPT_PDLP_SOLVER_MODE` via `cuOptSetIntegerParameter <lp-milp-c-api.html#c.cuOptSetIntegerParameter>`_.

.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_STABLE1
.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_STABLE2
.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_METHODICAL1
.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_FAST1

Method Constants
----------------

These constants are used to configure `CUOPT_METHOD` via `cuOptSetIntegerParameter <lp-milp-c-api.html#c.cuOptSetIntegerParameter>`_.

.. doxygendefine:: CUOPT_METHOD_CONCURRENT
.. doxygendefine:: CUOPT_METHOD_PDLP
.. doxygendefine:: CUOPT_METHOD_DUAL_SIMPLEX


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
.. doxygenfunction:: cuOptGetPrimalSolution
.. doxygenfunction:: cuOptGetObjectiveValue
.. doxygenfunction:: cuOptGetSolveTime
.. doxygenfunction:: cuOptGetMIPGap
.. doxygenfunction:: cuOptGetSolutionBound
.. doxygenfunction:: cuOptGetDualSolution
.. doxygenfunction:: cuOptGetReducedCosts

When you are finished with a `cuOptSolution` object you should destory it with

.. doxygenfunction:: cuOptDestroySolution

Termination Status Constants
----------------------------

These constants define the termination status received from the `cuOptGetTerminationStatus <lp-milp-c-api.html#c.cuOptGetTerminationStatus>`_ function.

.. LP/MIP termination status constants
.. doxygendefine:: CUOPT_TERIMINATION_STATUS_NO_TERMINATION
.. doxygendefine:: CUOPT_TERIMINATION_STATUS_OPTIMAL
.. doxygendefine:: CUOPT_TERIMINATION_STATUS_INFEASIBLE
.. doxygendefine:: CUOPT_TERIMINATION_STATUS_UNBOUNDED
.. doxygendefine:: CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT
.. doxygendefine:: CUOPT_TERIMINATION_STATUS_TIME_LIMIT
.. doxygendefine:: CUOPT_TERIMINATION_STATUS_NUMERICAL_ERROR
.. doxygendefine:: CUOPT_TERIMINATION_STATUS_PRIMAL_FEASIBLE
.. doxygendefine:: CUOPT_TERIMINATION_STATUS_FEASIBLE_FOUND
.. doxygendefine:: CUOPT_TERIMINATION_STATUS_CONCURRENT_LIMIT
