cuOpt MIP C API Reference
=========================

This section contains the cuOpt MIP C API reference. Functions for problem creation, solver settings, solving, and inspecting solutions are shared with convex optimization and documented in :doc:`../convex/convex-c-api`.

Warm Start and MIP Start
------------------------

For LP problems solved with PDLP, see :doc:`../convex/convex-c-api` for primal and dual warm start.

For MIP problems, one or more primal solution hints (MIP starts) may be provided:

.. doxygenfunction:: cuOptAddMIPStart

MIP Solution Callbacks
-----------------------

The following callback types and functions allow monitoring and injecting solutions during a MIP solve.

.. doxygentypedef:: cuOptMIPGetSolutionCallback
.. doxygentypedef:: cuOptMIPSetSolutionCallback

.. doxygenfunction:: cuOptSetMIPGetSolutionCallback
.. doxygenfunction:: cuOptSetMIPSetSolutionCallback

.. _mip-parameter-constants:

MIP Parameter Constants
-----------------------

These constants configure MIP-specific solver behavior. Use them with :c:func:`cuOptSetParameter`, :c:func:`cuOptSetIntegerParameter`, or :c:func:`cuOptSetFloatParameter`. For shared parameters (time limit, logging, presolve, etc.), see :ref:`parameter-constants` in the convex API reference.

.. doxygendefine:: CUOPT_NODE_LIMIT
.. doxygendefine:: CUOPT_WORK_LIMIT
.. doxygendefine:: CUOPT_RANDOM_SEED
.. doxygendefine:: CUOPT_PRESOLVE_FILE
.. doxygendefine:: CUOPT_MIP_ABSOLUTE_TOLERANCE
.. doxygendefine:: CUOPT_MIP_RELATIVE_TOLERANCE
.. doxygendefine:: CUOPT_MIP_INTEGRALITY_TOLERANCE
.. doxygendefine:: CUOPT_MIP_ABSOLUTE_GAP
.. doxygendefine:: CUOPT_MIP_RELATIVE_GAP
.. doxygendefine:: CUOPT_MIP_SCALING
.. doxygendefine:: CUOPT_MIP_HEURISTICS_ONLY
.. doxygendefine:: CUOPT_MIP_PRESOLVE
.. doxygendefine:: CUOPT_MIP_DETERMINISM_MODE
.. doxygendefine:: CUOPT_MIP_SYMMETRY
.. doxygendefine:: CUOPT_MIP_PROBING
.. doxygendefine:: CUOPT_MIP_RELIABILITY_BRANCHING
.. doxygendefine:: CUOPT_MIP_CUT_PASSES
.. doxygendefine:: CUOPT_MIP_MIXED_INTEGER_ROUNDING_CUTS
.. doxygendefine:: CUOPT_MIP_MIXED_INTEGER_GOMORY_CUTS
.. doxygendefine:: CUOPT_MIP_KNAPSACK_CUTS
.. doxygendefine:: CUOPT_MIP_FLOW_COVER_CUTS
.. doxygendefine:: CUOPT_MIP_IMPLIED_BOUND_CUTS
.. doxygendefine:: CUOPT_MIP_CLIQUE_CUTS
.. doxygendefine:: CUOPT_MIP_ZERO_HALF_CUTS
.. doxygendefine:: CUOPT_MIP_STRONG_CHVATAL_GOMORY_CUTS
.. doxygendefine:: CUOPT_MIP_REDUCED_COST_STRENGTHENING
.. doxygendefine:: CUOPT_MIP_OBJECTIVE_STEP
.. doxygendefine:: CUOPT_MIP_CUT_CHANGE_THRESHOLD
.. doxygendefine:: CUOPT_MIP_CUT_MIN_ORTHOGONALITY
.. doxygendefine:: CUOPT_MIP_BATCH_PDLP_STRONG_BRANCHING
.. doxygendefine:: CUOPT_MIP_BATCH_PDLP_RELIABILITY_BRANCHING
.. doxygendefine:: CUOPT_MIP_STRONG_BRANCHING_SIMPLEX_ITERATION_LIMIT
.. doxygendefine:: CUOPT_MIP_SEMICONTINUOUS_BIG_M

.. _mip-determinism-mode-constants:

MIP Determinism Mode Constants
------------------------------

These constants are used to configure `CUOPT_MIP_DETERMINISM_MODE` via :c:func:`cuOptSetIntegerParameter`.

.. doxygendefine:: CUOPT_MODE_OPPORTUNISTIC
.. doxygendefine:: CUOPT_MODE_DETERMINISTIC
