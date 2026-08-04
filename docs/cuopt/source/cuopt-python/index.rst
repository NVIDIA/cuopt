========================================
Python API
========================================

NVIDIA cuOpt supports Python API for routing optimization, convex optimization, and mixed-integer programming.

This section contains details on the cuOpt Python package.

For remote solves with **no code changes**, set ``CUOPT_REMOTE_HOST`` /
``CUOPT_REMOTE_PORT`` (see :doc:`../cuopt-grpc/quick-start`). For an
**explicit** Python job API against ``cuopt_grpc_server``, see the
:doc:`Python async gRPC client <../cuopt-grpc/python-async-client>`.

.. toctree::
   :maxdepth: 3
   :caption: Python API Overview
   :name: Python API Overview
   :titlesonly:

   quick-start.rst


.. toctree::
   :maxdepth: 3
   :caption: Routing Optimization
   :name: Routing Optimization
   :titlesonly:

   Routing Optimization <routing/index.rst>


.. toctree::
   :maxdepth: 3
   :caption: Convex Optimization (LP/QP/QCQP/SOCP)
   :name: Convex Optimization Python API
   :titlesonly:

   Convex Optimization <convex/index.rst>

.. toctree::
   :maxdepth: 3
   :caption: Mixed Integer Programming (MIP)
   :name: MIP Python API Index
   :titlesonly:

   MIP <mip/index.rst>
