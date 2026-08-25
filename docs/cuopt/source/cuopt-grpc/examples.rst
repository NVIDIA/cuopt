..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

========
Examples
========

**Remote execution** uses the same **Python**, **C API**, and **cuopt_cli**
entry points as a local solve. After you start ``cuopt_grpc_server`` on the
GPU server (:doc:`quick-start`), set the client environment and run the
integrated examples below **unchanged** — no code edits are required. The
:ref:`Python async gRPC client <cuopt-grpc-examples-async-client>` section is
separate: it uses ``Client(host, port)`` and does not read ``CUOPT_REMOTE_*``.

On the **client** machine, before running the example commands or scripts:

.. code-block:: bash

   export CUOPT_REMOTE_HOST=<gpu-server-hostname-or-ip>
   export CUOPT_REMOTE_PORT=5001

Add TLS or tuning variables from :doc:`advanced` if your deployment uses them.

.. note::

   Routing solve over gRPC is not supported. For solving routing problems remotely today, use the HTTP/JSON :doc:`REST self-hosted server <../cuopt-server/index>` and :doc:`Examples <../cuopt-server/examples/index>`.

Where to Find Examples
======================

Python (LP / QP / MIP)
-----------------------

* :doc:`../cuopt-python/convex/convex-examples` — runnable Python samples (LP, QP). With ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT`` set on the client, solves go to the remote server automatically.
* :doc:`../cuopt-python/mip/mip-examples` — runnable Python samples (MIP). With ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT`` set on the client, solves go to the remote server automatically.

C API (LP / QP / MIP)
----------------------

* :doc:`../cuopt-c/convex/convex-examples` — LP and QP C examples.
* :doc:`../cuopt-c/mip/mip-examples` — MIP C examples.

  Compile and run these programs with the same exports in the shell;
  ``cuOptSolve`` uses gRPC when both remote variables are set (see
  :doc:`../cuopt-c/convex/convex-c-api` for API reference).

``cuopt_cli``
-------------

* :doc:`../cuopt-cli/cli-examples` — ``cuopt_cli`` invocations. With the exports above, the CLI forwards solves to ``cuopt_grpc_server``.

Minimal Demos (This Section)
----------------------------

Included with the gRPC docs source for a quick copy-paste path (also walked through in :doc:`quick-start`):

* :download:`remote_lp_demo.py <examples/remote_lp_demo.py>`
* :download:`remote_lp_demo.mps <examples/remote_lp_demo.mps>`

Python Async gRPC Client
------------------------

.. _cuopt-grpc-examples-async-client:

For explicit job control (submit / wait / cancel / stream logs or incumbents)
without ``CUOPT_REMOTE_*``, use ``cuopt.grpc.linear_programming.Client``:

* :doc:`python-async-client` — overview
* :doc:`python-async-client-examples` — log streaming and incumbent streaming
* :doc:`python-async-client-api` — API reference

Custom gRPC Client
------------------

Integrations that do **not** use remote execution or the Python async gRPC
client should speak ``CuOptRemoteService`` directly. See :doc:`api`,
:doc:`grpc-server-architecture`, and ``cpp/docs/grpc-server-architecture.md``
in the repository for protos and server behavior.

More Samples
============

* `NVIDIA cuOpt examples on GitHub <https://github.com/NVIDIA/cuopt-examples>`_ — set the remote environment on the **client** before running notebooks or scripts.

REST vs gRPC
============

* **Self-hosted HTTP/JSON** — :doc:`../cuopt-server/examples/index` targets the REST server; request shapes follow the OpenAPI workflow, not the ``CuOptRemoteService`` protos.
