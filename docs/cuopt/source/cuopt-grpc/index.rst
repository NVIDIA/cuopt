..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

==========================
gRPC remote execution
==========================

**NVIDIA cuOpt gRPC remote execution** runs optimization solves on a remote GPU host. Clients can be the **Python** API, **C API**, **`cuopt_cli`**, or a **custom** program that speaks ``CuOptRemoteService`` over gRPC. For Python, the C API, and ``cuopt_cli``, set ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT`` to forward solves to ``cuopt_grpc_server``.

.. note::

   **Problem types (gRPC remote):** LP, MILP, and QP are supported today. **Routing** (VRP, TSP, PDP, and related APIs) over gRPC remote execution is **not** available yet; support is planned for an **upcoming** release. For routing against a remote service today, use the HTTP/JSON :doc:`REST self-hosted server <../cuopt-server/index>`.

This is **not** the HTTP/JSON :doc:`REST self-hosted server <../cuopt-server/index>` (FastAPI). REST is for arbitrary HTTP clients; gRPC is for the bundled remote client in NVIDIA cuOpt's native APIs.

Start with :doc:`quick-start` (install selector, how remote execution works, Docker, and a minimal LP example). Use :doc:`advanced` for TLS, tuning, limitations, and troubleshooting; :doc:`examples` for additional patterns.

.. toctree::
   :maxdepth: 2
   :caption: In this section
   :name: cuopt-grpc-contents

   quick-start.rst
   advanced.rst
   examples.rst
   api.rst
   grpc-server-architecture.md

See :doc:`../system-requirements` for GPU, CUDA, and OS requirements.
