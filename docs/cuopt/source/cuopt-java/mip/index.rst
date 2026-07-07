=====================================
Mixed Integer Programming (MIP)
=====================================

.. note::

   The cuOpt MIP solver is in beta and under active development. Finding
   high-quality feasible solutions and proving optimality have different
   performance characteristics; inspect termination status, MIP gap, and
   solution bound in production workflows.

The Java MIP API uses the same ``Problem``, ``DataModel``,
``SolverSettings``, and ``Solution`` classes as the LP/QP API. MIP models are
identified by one or more variables with ``INTEGER`` or ``SEMI_CONTINUOUS``
type.

.. toctree::
   :maxdepth: 3
   :caption: MIP Java API
   :name: MIP Java API Reference
   :titlesonly:

   mip-api.rst
   mip-examples.rst
