====================================
Java API (Beta)
====================================

NVIDIA cuOpt provides experimental Java bindings for linear programming (LP),
mixed-integer linear programming (MILP), quadratic programming (QP), and
quadratic constraints through JNI.

The Java bindings are a separately compiled beta module for the LP/MILP/QP
surface. They are not part of the repository-level cuOpt build and do not
provide routing or distance-engine bindings. See :doc:`quick-start` before
using the API.

.. note::

   This Java module is currently intended for the customer beta described by
   your cuOpt distribution. Build it from ``java/cuopt`` against an existing
   cuOpt installation; do not expect ``build.sh`` to compile it.

.. toctree::
   :maxdepth: 3
   :caption: Java API Overview
   :name: Java API Overview
   :titlesonly:

   quick-start.rst

.. toctree::
   :maxdepth: 3
   :caption: Convex Optimization (LP/QP)
   :name: LP/QP Java API
   :titlesonly:

   Convex Optimization <convex/index.rst>

.. toctree::
   :maxdepth: 3
   :caption: Mixed Integer Programming (MIP)
   :name: MIP Java API
   :titlesonly:

   Mixed Integer Programming <mip/index.rst>
