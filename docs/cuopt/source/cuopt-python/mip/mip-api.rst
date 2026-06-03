=================
MIP API Reference
=================

MIP uses the same Python API classes as convex optimization — see :doc:`../convex/convex-api` for the full class reference. The following are particularly relevant for MIP problems.

Variable Types
--------------

Use :class:`~cuopt.linear_programming.problem.VType` to declare integer and semi-continuous variables:

.. autoclass:: cuopt.linear_programming.problem.VType
   :members:
   :member-order: bysource
   :undoc-members:
   :exclude-members: capitalize, casefold, center, count, encode, endswith, expandtabs, find, format, format_map, index, isalnum, isalpha, isascii, isdecimal, isdigit, isidentifier, islower, isnumeric, isprintable, isspace, istitle, isupper, join, ljust, lower, lstrip, maketrans, partition, removeprefix, removesuffix, replace, rfind, rindex, rjust, rpartition, rsplit, rstrip, split, splitlines, startswith, strip, swapcase, title, translate, upper, zfill

MIP Start
---------

:class:`~cuopt.linear_programming.problem.Problem` supports providing initial feasible solutions (MIP starts) to help the solver find good solutions faster. Use ``Problem.addMIPStart`` to add one or more primal solution hints before calling ``solve()``.

MIP Solution Callbacks
-----------------------

:class:`~cuopt.linear_programming.solver_settings.SolverSettings` supports callbacks to monitor incumbent solutions found during the MIP solve. Use ``SolverSettings.set_mip_solution_callback`` to register a callback that receives each new incumbent solution.
