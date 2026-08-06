..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

==========================
gRPC Remote Execution
==========================

NVIDIA cuOpt can run LP, MIP, and QP solves on a remote GPU host through
``cuopt_grpc_server``. There are two ways to reach that server:

**Remote execution** (zero code change)
  Set ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT`` on the **client** machine.
  The Python solver APIs, the C API ``cuOptSolve``, and ``cuopt_cli`` forward
  the solve automatically. No client code changes are required.
  See :doc:`quick-start`.

**gRPC clients** (explicit client)
  Your program opens a gRPC connection and manages jobs itself. Use the
  :doc:`Python async gRPC client <python-async-client>`
  (``cuopt.grpc.linear_programming.Client``) for job management, or speak
  ``CuOptRemoteService`` directly from a custom client (:doc:`api`).

In this section, **remote execution** always means the zero-code-change path
above. When talking about programs that construct a client and call gRPC
themselves, we say **gRPC client**.

.. note::

   **Problem types:** LP, MIP, and QP are supported today. **Routing** (VRP,
   TSP, PDP, and related APIs) over gRPC is **not** available yet; support is
   planned for an **upcoming** release. For remote routing today, use the
   HTTP/JSON :doc:`REST self-hosted server <../cuopt-server/index>`.

This is **not** the HTTP/JSON :doc:`REST self-hosted server <../cuopt-server/index>`
(FastAPI). REST is for arbitrary HTTP clients; gRPC serves remote execution
(client integrated into the solver APIs) and explicit gRPC clients.

When to Choose Which Path
=========================

* **Remote execution** — drop-in remote solves with no code changes; same
  scripts and APIs as a local solve.
* **Python async gRPC client** — explicit job control: submit now, wait or
  poll later, cancel, stream solver logs, stream MIP incumbents.
* **Custom ``CuOptRemoteService`` client** — non-Python (or fully custom)
  integrations that speak the protos directly. See :doc:`api`.

Start with :doc:`quick-start` (install, server, and a minimal LP). Use
:doc:`python-async-client` for the Python gRPC client; :doc:`advanced` for
TLS, Docker, environment variables, and troubleshooting; :doc:`examples` for
additional patterns.

.. toctree::
   :maxdepth: 2
   :caption: In this section
   :name: cuopt-grpc-contents

   quick-start.rst
   python-async-client.rst
   python-async-client-examples.rst
   python-async-client-api.rst
   advanced.rst
   examples.rst
   api.rst
   grpc-server-architecture.md

See :doc:`../system-requirements` for GPU, CUDA, and OS requirements.
