..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

====
gRPC
====

NVIDIA cuOpt can run LP, MIP, QP, and routing (VRP, TSP, PDP) solves on a
remote GPU host through ``cuopt_grpc_server``. There are two ways to reach
that server:

**Remote execution** (zero code change)
  Set ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT`` on the **client** machine.
  The Python solver APIs, the C API ``cuOptSolve``, and ``cuopt_cli`` forward
  LP, MIP, and QP solves automatically. Modeling-language integrations that
  call one of those APIs, including AMPL, GAMS, PuLP, JuMP, Pyomo, and CVXPY,
  forward the same way. No client code changes are required. This does not
  work for routing at this time (tracked in `#1633
  <https://github.com/NVIDIA/cuopt/issues/1633>`_).
  See :doc:`quick-start` and
  :doc:`../thirdparty_modeling_languages/index`.

**gRPC clients** (explicit client)
  Your program opens a gRPC connection and manages jobs itself. cuOpt ships
  two Python clients. Both are asynchronous job APIs: submit, then wait or
  poll.

  * The :doc:`gRPC linear programming client <python-async-client>`
    (``cuopt.grpc.linear_programming.Client``) manages LP, MIP, and QP jobs.
  * The :doc:`gRPC routing client <routing>`
    (``cuopt.grpc.routing.RoutingClient``) manages VRP, TSP, and PDP jobs.
  * A custom client can speak ``CuOptRemoteService`` directly (:doc:`api`).

In this section, **remote execution** always means the zero-code-change path
above. **gRPC linear programming client** and **gRPC routing client** are the
two Python clients. **gRPC client** means any program that opens the
connection itself, including a custom ``CuOptRemoteService`` client.

When to Choose Which Path
=========================

* **Remote execution** — drop-in remote LP, MIP, and QP solves with no code
  changes, including modeling-language integrations that call the Python
  solver APIs, ``cuOptSolve``, or ``cuopt_cli``.
* **gRPC linear programming client** — explicit LP, MIP, and QP job control:
  submit now, wait or poll later, cancel, stream solver logs, stream MIP
  incumbents.
* **gRPC routing client** — explicit VRP, TSP, and PDP job control: submit
  now, wait or poll later, cancel.
* **Custom ``CuOptRemoteService`` client** — non-Python (or fully custom)
  integrations that speak the protos directly. See :doc:`api`.

Start with :doc:`quick-start` (install, server, and a minimal LP). Use
:doc:`python-async-client` for the gRPC linear programming client and
:doc:`routing` for the gRPC routing client. See :doc:`advanced` for TLS,
Docker, environment variables, and troubleshooting, and :doc:`examples` for
additional patterns.

.. toctree::
   :maxdepth: 2
   :caption: In this section
   :name: cuopt-grpc-contents

   quick-start.rst
   python-async-client.rst
   python-async-client-examples.rst
   python-async-client-api.rst
   routing.rst
   routing-api.rst
   advanced.rst
   examples.rst
   api.rst
   grpc-server-architecture.md

See :doc:`../system-requirements` for GPU, CUDA, and OS requirements.
