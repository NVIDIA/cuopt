..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

========
Examples
========

gRPC remote execution uses the same **Python**, **C API**, and **cuopt_cli** entry points as a local solve. After you start ``cuopt_grpc_server`` on the GPU host (:doc:`quick-start`), set the client environment and run **any** of the examples below **unchanged** — no code edits are required.

On the **client** host, before running the example commands or scripts:

.. code-block:: bash

   export CUOPT_REMOTE_HOST=<gpu-hostname-or-ip>
   export CUOPT_REMOTE_PORT=5001

Add TLS or tuning variables from :doc:`advanced` if your deployment uses them.

.. note::

   Routing solve over gRPC is not supported. For solving routing problems remotely today, use the HTTP/JSON :doc:`REST self-hosted server <../cuopt-server/index>` and :doc:`Examples <../cuopt-server/examples/index>`.

Where to find examples
======================

Python (LP / QP / MILP)
-----------------------

* :doc:`../cuopt-python/lp-qp-milp/lp-qp-milp-examples` — runnable Python samples (LP, QP, MILP). With ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT`` set on the client, solves go to the remote server automatically.

C API (LP / QP / MILP)
----------------------

* :doc:`../cuopt-c/lp-qp-milp/lp-qp-example` — LP and QP C examples.
* :doc:`../cuopt-c/lp-qp-milp/milp-examples` — MILP C examples.

  Compile and run these programs with the same exports in the shell; ``solve_lp`` / ``solve_mip`` use gRPC when both remote variables are set (see :doc:`../cuopt-c/lp-qp-milp/lp-qp-milp-c-api` for API reference).

``cuopt_cli``
-------------

* :doc:`../cuopt-cli/cli-examples` — ``cuopt_cli`` invocations. With the exports above, the CLI forwards solves to ``cuopt_grpc_server``.

Minimal demos (this section)
----------------------------

Bundled with the gRPC docs source for a quick copy-paste path (also walked through in :doc:`quick-start`):

* :download:`remote_lp_demo.py <examples/remote_lp_demo.py>`
* :download:`remote_lp_demo.mps <examples/remote_lp_demo.mps>`

Custom gRPC client
------------------

Integrations that do **not** use the bundled Python / C / CLI stack should speak ``CuOptRemoteService`` directly. See :doc:`api`, :doc:`grpc-server-architecture`, and ``cpp/docs/grpc-server-architecture.md`` in the repository for protos and server behavior.

More samples
============

* `NVIDIA cuOpt examples on GitHub <https://github.com/NVIDIA/cuopt-examples>`_ — set the remote environment on the **client** before running notebooks or scripts.

REST vs gRPC
============

* **Self-hosted HTTP/JSON** — :doc:`../cuopt-server/examples/index` targets the REST server; request shapes follow the OpenAPI workflow, not the ``CuOptRemoteService`` protos.
