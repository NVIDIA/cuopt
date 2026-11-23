..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

=================================
gRPC Routing Client API Reference
=================================

Import path: ``cuopt.grpc.routing``.

Client
======

.. autoclass:: cuopt.grpc.routing.RoutingClient
   :members:
   :undoc-members:

Supporting Types
================

.. Same class as :class:`cuopt.grpc.linear_programming.JobStatus`, re-exported
   for routing.

.. autoclass:: cuopt.grpc.routing.JobStatus
   :no-index:
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: __new__, __init__, _generate_next_value_, as_integer_ratio, bit_count, bit_length, conjugate, denominator, from_bytes, imag, is_integer, numerator, real, to_bytes

Exceptions
==========

.. autoexception:: cuopt.grpc.routing.RoutingSolveError
   :members:
   :show-inheritance:

.. _routing-api-settings:

Settings
========

``settings`` accepts a ``dict`` or a :class:`cuopt.routing.SolverSettings`.
``time_limit``, ``verbose_mode`` (a ``dict`` may use ``verbose``), and
``error_logging`` are forwarded to the remote solve. ``dump_best_results``
is not forwarded: the solver would write that file on the server.
``dump_config_file`` is not forwarded either. Local
:func:`cuopt.routing.Solve` writes that YAML in the calling process after
the solve; the remote solve does not. Any other ``dict`` key is silently
ignored.

.. _routing-api-solution-fields:

Solution Fields
===============

``result()`` returns ``None`` while the job is still queued or running.
Once a solution is available, it returns a ``dict`` read from the routing
solution message. ``solve()`` returns that ``dict`` when the job completes,
and raises ``RoutingSolveError`` otherwise. Route data is one NumPy array
per field, with one entry per stop.

.. list-table::
   :header-rows: 1

   * - Key
     - Description
   * - ``status``
     - ``0`` success, ``1`` infeasible, ``2`` timeout, ``3`` empty, ``4`` error.
   * - ``status_message``
     - Solver status text, such as ``Success``.
   * - ``error_message``
     - Set when ``status`` is ``4``. Empty otherwise.
   * - ``solve_time``
     - Seconds spent in the routing solver on the server.
   * - ``vehicle_count``
     - Number of vehicles used.
   * - ``total_objective_value``
     - Overall cost.
   * - ``objective_values``
     - ``dict`` of objective id to cost.
   * - ``route``, ``truck_id``, ``locations``, ``node_types``
     - Per-stop integer arrays.
   * - ``arrival_stamp``
     - Per-stop arrival times.
   * - ``unserviced_nodes``
     - Orders that cannot be served.
   * - ``accepted``
     - Orders accepted, for prize-collection problems.

See Also
========

* :doc:`routing` — overview and the connect-and-solve example
* :doc:`api` — ``CuOptRemoteService`` proto / RPC reference
* :doc:`../cuopt-python/routing/index` — the local routing Python API
