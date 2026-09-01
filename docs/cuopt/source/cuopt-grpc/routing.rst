..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

=============================
VRP gRPC Client (Routing)
=============================

``cuopt.grpc.routing.RoutingClient`` is an explicit gRPC client for solving
**VRP** (vehicle routing, including TSP and PDP) problems on
``cuopt_grpc_server``. It uses the same job lifecycle as the LP/MIP
:doc:`Python async gRPC client <python-async-client>`: **submit** → **wait**
→ **result** → **delete**, plus a **solve** convenience method that does all
four.

There is no ``CUOPT_REMOTE_HOST``/``CUOPT_REMOTE_PORT`` transparent path for
routing yet (unlike LP/MIP/QP) -- always construct ``RoutingClient`` with an
explicit host and port. See :ref:`Limitations and Roadmap
<cuopt-grpc-routing-limitations>` below.

Prerequisites
=============

A running ``cuopt_grpc_server`` on a GPU host (see :doc:`quick-start`):

.. code-block:: bash

   cuopt_grpc_server --port 5001 --workers 1

Connect and Solve
==================

.. code-block:: python

   import numpy as np
   from cuopt import routing
   from cuopt.grpc.routing import RoutingClient

   dm = routing.DataModel(5, 2)
   cost_matrix = np.array(
       [
           [0, 1, 2, 2, 1],
           [1, 0, 1, 2, 2],
           [2, 1, 0, 1, 2],
           [2, 2, 1, 0, 1],
           [1, 2, 2, 1, 0],
       ],
       dtype=np.float32,
   )
   dm.add_cost_matrix(cost_matrix)

   client = RoutingClient("localhost:5001")
   solution = client.solve(dm, {"time_limit": 5.0})

   print(solution["status_message"])
   print(solution["total_objective_value"])
   print(solution["route"])

``RoutingClient.submit()`` accepts a :class:`cuopt.routing.DataModel` built
the same way as for a local :func:`cuopt.routing.Solve`. ``solve()`` submits,
waits, and deletes the job's server-side state when done (pass
``delete=False`` to keep it around for a later ``result()`` call).

:download:`remote_routing_demo.py <examples/remote_routing_demo.py>`

.. literalinclude:: examples/remote_routing_demo.py
   :language: python
   :linenos:

Job Lifecycle
=============

* ``submit(data_model, settings=None)`` — serializes the problem and settings, returns a ``job_id``.
* ``wait(job_id, timeout=0)`` — blocks until the job reaches a terminal state; returns the status.
* ``result(job_id)`` — returns the solution dict, or ``None`` if the job has not finished.
* ``delete(job_id)`` — releases the job's server-side result.
* ``solve(data_model, settings=None, *, timeout=0, delete=True)`` — submit + wait + result, deleting the job afterward unless ``delete=False``.

A failed or non-completed job raises ``RoutingSolveError`` from ``submit``,
``wait``, or ``solve``.

Settings
========

``settings`` accepts a ``dict`` or a :class:`cuopt.routing.SolverSettings`.
Today only ``time_limit`` is forwarded to the remote solve; other
``SolverSettings`` options (``verbose``, ``error_logging``,
``dump_best_results_path``/``interval``) are not yet mapped over gRPC (see
:ref:`Limitations and Roadmap <cuopt-grpc-routing-limitations>`).

Solution Fields
================

``result()`` and ``solve()`` return a ``dict`` with the same fields as a
local :class:`cuopt.routing.Assignment`, read directly off the wire:

.. list-table::
   :header-rows: 1

   * - Key
     - Description
   * - ``status`` / ``status_message``
     - Integer and human-readable solve status.
   * - ``error_message``
     - Set when the solve failed.
   * - ``vehicle_count``
     - Number of vehicles used.
   * - ``total_objective_value`` / ``objective_values``
     - Overall cost and the per-objective breakdown.
   * - ``route``, ``truck_id``, ``locations``, ``node_types``, ``arrival_stamp``
     - Per-stop route arrays, one entry per stop across all vehicles.
   * - ``unserviced_nodes``
     - Orders that could not be served.
   * - ``accepted``
     - Orders accepted, for prize-collection problems.

.. _cuopt-grpc-routing-limitations:

Limitations and Roadmap
=========================

* **No transparent remote execution** — routing does not read
  ``CUOPT_REMOTE_HOST``/``CUOPT_REMOTE_PORT``; always pass host and port to
  ``RoutingClient`` explicitly. Tracked in `#1633
  <https://github.com/NVIDIA/cuopt/issues/1633>`_.
* **Settings surface** — only ``time_limit`` is forwarded today. Tracked in
  `#1632 <https://github.com/NVIDIA/cuopt/issues/1632>`_.
* **No log or incumbent streaming** — unlike the LP/MIP client, there is no
  ``start_log_stream``/``start_incumbent_stream`` equivalent yet. Tracked in
  `#1630 <https://github.com/NVIDIA/cuopt/issues/1630>`_.
* **Input validation** — malformed problems may fail late or with a generic
  error rather than an early, descriptive one. Tracked in `#1631
  <https://github.com/NVIDIA/cuopt/issues/1631>`_.

API Reference
=============

Import path: ``cuopt.grpc.routing``.

.. autoclass:: cuopt.grpc.routing.RoutingClient
   :members:
   :undoc-members:

.. autoexception:: cuopt.grpc.routing.RoutingSolveError
   :members:
   :show-inheritance:

See Also
========

* :doc:`index` — when to use gRPC vs. the REST self-hosted server
* :doc:`python-async-client` — the LP/MIP/QP equivalent client
* :doc:`api` — how VRP rides ``CuOptRemoteService``'s RPCs
* :doc:`../cuopt-python/routing/index` — the local routing Python API
