..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

====================
gRPC Routing Client
====================

``cuopt.grpc.routing.RoutingClient`` is the **gRPC routing client**, a Python
client for **VRP** (vehicle routing, including TSP and PDP) jobs on
``cuopt_grpc_server``. It uses a job lifecycle: **submit** → **wait** /
**status** → **result** → **delete**, plus a **solve** convenience method
that does all four.

Asynchronous here means the job-based API (submit, then wait or poll). It is
separate from Python ``asyncio``.

There is no ``CUOPT_REMOTE_HOST``/``CUOPT_REMOTE_PORT`` transparent path for
routing at this time (unlike LP/MIP/QP) -- always construct ``RoutingClient`` with an
explicit host and port. See :ref:`Limitations and Roadmap
<cuopt-grpc-routing-limitations>` below.

Prerequisites
=============

A running ``cuopt_grpc_server`` on a GPU host (see :doc:`quick-start`):

.. code-block:: bash

   cuopt_grpc_server --port 5001 --workers 1

Connect and Solve
==================

``RoutingClient(host, port, *, tls=None)`` takes the same arguments as the
LP/MIP client's ``Client(host, port, tls=...)``; see :doc:`python-async-client`
for the ``tls`` options.

``RoutingClient.submit()`` accepts a :class:`cuopt.routing.DataModel` built
the same way as for a local :func:`cuopt.routing.Solve`. ``solve()`` submits,
waits, and deletes the job's server-side state when done (pass
``delete=False`` to keep it around for a later ``result()`` call).

:download:`remote_routing_demo.py <examples/remote_routing_demo.py>`

.. literalinclude:: examples/remote_routing_demo.py
   :language: python
   :linenos:

Illustrative output (exact status text, vehicle count, objective, and route
order depend on the solver run and are not guaranteed to match):

.. code-block:: text

   Status:     Success
   Vehicles:   1
   Objective:  5.0
   Route:      [0 1 2 3 4 0]

``settings`` accepts a ``dict`` or a :class:`cuopt.routing.SolverSettings`.
The :doc:`API reference <routing-api>` describes which settings are forwarded
and what ``result()`` and ``solve()`` return.

.. _cuopt-grpc-routing-limitations:

Limitations and Roadmap
=========================

* **No transparent remote execution** — routing does not read
  ``CUOPT_REMOTE_HOST``/``CUOPT_REMOTE_PORT``; always pass host and port to
  ``RoutingClient`` explicitly. Tracked in `#1633
  <https://github.com/NVIDIA/cuopt/issues/1633>`_.
* **Settings surface** — ``time_limit``, ``verbose_mode`` (or ``verbose``),
  and ``error_logging`` are forwarded. ``dump_best_results`` is not, because
  the solver would write that file on the server. ``dump_config_file`` is
  not forwarded; the local solve writes that YAML in the calling process.
  Tracked in `#1632 <https://github.com/NVIDIA/cuopt/issues/1632>`_.

Next Steps
==========

* :doc:`routing-api` — API reference
* :doc:`quick-start` — start ``cuopt_grpc_server``
* :doc:`api` — low-level ``CuOptRemoteService`` RPCs
* :doc:`../cuopt-python/routing/index` — the local routing Python API
