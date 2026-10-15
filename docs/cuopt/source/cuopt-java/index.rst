====================================
Java API
====================================

NVIDIA cuOpt provides experimental Java bindings for linear programming (LP),
mixed-integer programming (MIP), quadratic programming (QP),
quadratically constrained quadratic programming (QCQP), and second-order cone
programming (SOCP) through JNI.

The Java bindings are a separately compiled beta module for this mathematical
programming surface. Repository CI and release workflows build and test it
against the matching ``libcuopt`` artifact, but it is not part of the top-level
cuOpt build and does not provide routing or distance-engine bindings. See
:doc:`quick-start` before using the API.

.. note::

   Build the module locally from ``java/cuopt`` against an existing cuOpt
   installation. CI artifacts are experimental; publication to a supported
   Maven repository has not been defined.

.. toctree::
   :maxdepth: 3
   :caption: Java API Overview
   :name: Java API Overview
   :titlesonly:

   quick-start.rst

.. toctree::
   :maxdepth: 3
   :caption: Convex Optimization (LP/QP/QCQP/SOCP)
   :name: LP/QP Java API
   :titlesonly:

   Convex Optimization <convex/index.rst>

.. toctree::
   :maxdepth: 3
   :caption: Mixed Integer Programming (MIP)
   :name: MIP Java API
   :titlesonly:

   Mixed Integer Programming <mip/index.rst>
