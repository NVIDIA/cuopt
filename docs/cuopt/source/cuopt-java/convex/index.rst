================================================
Convex Optimization (LP/QP/QCQP/SOCP)
================================================

This section documents the Java bindings for LP, QP, QCQP, and SOCP. The Java
API includes both:

* a high-level problem API based on ``Problem``, ``Variable``, expressions,
  and constraints; and
* a deprecated lower-level ``DataModel`` API for compatibility with direct
  CSR data, ranged bounds, quadratic matrices, MPS I/O, and solver results.

Quadratic constraints are supported for ``LE`` and ``GE`` constraints. Equality
quadratic constraints are rejected by the Java API.

.. toctree::
   :maxdepth: 3
   :caption: LP/QP/QCQP/SOCP Java API
   :name: LP/QP/QCQP/SOCP Java API Reference
   :titlesonly:

   convex-api.rst
   convex-examples.rst
