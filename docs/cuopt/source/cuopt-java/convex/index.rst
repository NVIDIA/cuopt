=====================================
Convex Optimization (LP/QP)
=====================================

This section documents the Java bindings for continuous linear and quadratic
optimization. The Java API includes both:

* a high-level modeling API based on ``Problem``, ``Variable``, expressions,
  and constraints; and
* a lower-level ``DataModel`` API that exposes CSR data, ranged bounds,
  quadratic matrices, MPS I/O, and solver results.

Quadratic constraints are supported for ``LE`` and ``GE`` constraints. Equality
quadratic constraints are rejected by the Java API. Dedicated SOCP modeling
helpers are not currently exposed; cone models can be represented through the
supported quadratic-expression API when they satisfy cuOpt's quadratic
constraint requirements.

.. toctree::
   :maxdepth: 3
   :caption: LP/QP Java API
   :name: LP/QP Java API Reference
   :titlesonly:

   convex-api.rst
   convex-examples.rst
