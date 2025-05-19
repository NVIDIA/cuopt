cuOpt LP/MILP C API Reference
========================================

This section contains the cuOpt LP/MILP C API reference.

Types
-----

.. doxygentypedef:: cuOptOptimizationProblem
.. doxygentypedef:: cuOptSolverSettings
.. doxygentypedef:: cuOptSolution
.. doxygentypedef:: cuopt_float_t
.. doxygentypedef:: cuopt_int_t

Status Constants
----------------

.. Status code constants
.. doxygendefine:: CUOPT_SUCCESS
.. doxygendefine:: CUOPT_INVALID_ARGUMENT
.. doxygendefine:: CUOPT_MPS_FILE_ERROR
.. doxygendefine:: CUOPT_MPS_PARSE_ERROR

Parameter Constants
------------------- 

These constants would be used as the parameter name in the `cuOptSetParameter <lp-milp-c-api.html#c.cuOptSetParameter>`_ and `cuOptGetParameter <lp-milp-c-api.html#c.cuOptGetParameter>`_ functions. More details on the parameters can be found in the `LP/MILP settings <../../lp-milp-settings.html>`_ section.

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

Termination Status Constants
----------------------------

These constants would be used as the termination status in the `cuOptGetTerminationStatus <lp-milp-c-api.html#c.cuOptGetTerminationStatus>`_ function.

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

Objective Sense Constants
-------------------------

These would be used as the objective sense in the `cuOptCreateProblem <lp-milp-c-api.html#c.cuOptCreateProblem>`_ and `cuOptCreateRangedProblem <lp-milp-c-api.html#c.cuOptCreateRangedProblem>`_ functions.

.. doxygendefine:: CUOPT_MINIMIZE
.. doxygendefine:: CUOPT_MAXIMIZE

Constraint Sense Constants
--------------------------

These would be used as the constraint sense in the `cuOptCreateProblem <lp-milp-c-api.html#c.cuOptCreateProblem>`_ and `cuOptCreateRangedProblem <lp-milp-c-api.html#c.cuOptCreateRangedProblem>`_ functions.

.. doxygendefine:: CUOPT_LESS_THAN
.. doxygendefine:: CUOPT_GREATER_THAN
.. doxygendefine:: CUOPT_EQUAL

Variable Type Constants
-----------------------

These would be used as the variable type in the `cuOptCreateProblem <lp-milp-c-api.html#c.cuOptCreateProblem>`_ and `cuOptCreateRangedProblem <lp-milp-c-api.html#c.cuOptCreateRangedProblem>`_ functions.

.. doxygendefine:: CUOPT_CONTINUOUS
.. doxygendefine:: CUOPT_INTEGER

Infinity Constant
-----------------

This would be used as the infinity value in the `cuOptCreateProblem <lp-milp-c-api.html#c.cuOptCreateProblem>`_ and `cuOptCreateRangedProblem <lp-milp-c-api.html#c.cuOptCreateRangedProblem>`_ functions.

.. doxygendefine:: CUOPT_INFINITY

PDLP Solver Mode Constants
--------------------------

These would be used as the PDLP solver mode while setting solver parameters using `cuOptSetParameter <lp-milp-c-api.html#c.cuOptSetParameter>`_.

.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_STABLE1
.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_STABLE2
.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_METHODICAL1
.. doxygendefine:: CUOPT_PDLP_SOLVER_MODE_FAST1

Method Constants
----------------

These would be used as the method while setting solver parameters using `cuOptSetParameter <lp-milp-c-api.html#c.cuOptSetParameter>`_.

.. doxygendefine:: CUOPT_METHOD_CONCURRENT
.. doxygendefine:: CUOPT_METHOD_PDLP
.. doxygendefine:: CUOPT_METHOD_DUAL_SIMPLEX

Functions
---------

.. cuopt_c.h functions
.. doxygenfunction:: cuOptGetFloatSize
.. doxygenfunction:: cuOptGetIntSize
.. doxygenfunction:: cuOptReadProblem
.. doxygenfunction:: cuOptCreateProblem
.. doxygenfunction:: cuOptCreateRangedProblem
.. doxygenfunction:: cuOptDestroyProblem
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
.. doxygenfunction:: cuOptCreateSolverSettings
.. doxygenfunction:: cuOptDestroySolverSettings

More details on the parameters can be found in the `LP/MILP settings <../../lp-milp-settings.html>`_ section.

.. doxygenfunction:: cuOptSetParameter
.. doxygenfunction:: cuOptGetParameter
.. doxygenfunction:: cuOptSetIntegerParameter
.. doxygenfunction:: cuOptGetIntegerParameter
.. doxygenfunction:: cuOptSetFloatParameter
.. doxygenfunction:: cuOptGetFloatParameter
.. doxygenfunction:: cuOptIsMIP
.. doxygenfunction:: cuOptSolve
.. doxygenfunction:: cuOptDestroySolution
.. doxygenfunction:: cuOptGetTerminationStatus
.. doxygenfunction:: cuOptGetPrimalSolution
.. doxygenfunction:: cuOptGetObjectiveValue
.. doxygenfunction:: cuOptGetSolveTime
.. doxygenfunction:: cuOptGetMIPGap
.. doxygenfunction:: cuOptGetSolutionBound
.. doxygenfunction:: cuOptGetDualSolution
.. doxygenfunction:: cuOptGetReducedCosts

CLI for LP and MILP
===================

The cuOpt CLI is a command-line interface for the cuOpt LP/MILP API. It is a simple interface that allows you to solve LP/MILP problems from the command line. This CLI is based on C/C++ API.

.. literalinclude:: cuopt-cli-help.txt
   :language: shell
   :linenos:

