..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

===========
Quick start
===========

**NVIDIA cuOpt gRPC remote execution** runs LP, MILP, and QP solves on a **GPU host** while your **Python** code, **C API** program, **`cuopt_cli`**, or a **custom** client runs elsewhere. When you set ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT``, the bundled **Python**, **C API**, and **cuopt_cli** clients forward ``solve_lp`` / ``solve_mip`` to ``cuopt_grpc_server`` with **no code changes**. **Custom** clients call ``CuOptRemoteService`` directly (see :doc:`api`).

.. note::

   **Problem types (gRPC remote):** **LP**, **MILP**, and **QP** are supported today. **Routing** (VRP, TSP, PDP) over this path is **not** available;  For remote routing, use the HTTP/JSON :doc:`REST self-hosted server <../cuopt-server/index>`. This guide is **not** the REST server—see :doc:`../cuopt-server/index` for HTTP/JSON.

How remote execution works
==========================

1. **GPU host** — Run ``cuopt_grpc_server`` (bare metal or in the official container) so it listens on a TCP port (default **5001**).
2. **Client** — Install the NVIDIA cuOpt client libraries on the machine where you invoke the solver. Set ``CUOPT_REMOTE_HOST`` to that GPU host’s address and ``CUOPT_REMOTE_PORT`` to the listen port.
3. **Solve** — Call the same APIs you would for a local solve. The client library opens a gRPC channel, streams the problem, and retrieves the result. Unset the two variables to solve **locally** again (local mode still needs a GPU on that machine where applicable).

Install NVIDIA cuOpt
====================

Use the selector below on the **GPU server** and on **clients** that need Python, the C API, or ``cuopt_cli``. It is pre-set to **C (libcuopt)** because that bundle ships ``cuopt_grpc_server``, ``cuopt_cli``, and libraries together; switch to **Python** if you only need Python packages on a lightweight client.

.. install-selector::
   :default-iface: c

Verify the server binary after install:

.. code-block:: bash

   cuopt_grpc_server --help

For the same install selector with **Container** / registry choices (Docker Hub or NGC), see :doc:`../install`.

Run the gRPC server (GPU host)
==============================

**Bare metal** — after activating the same environment you used to install NVIDIA cuOpt:

.. code-block:: bash

   cuopt_grpc_server --port 5001 --workers 1

Leave the process running. Default port **5001**; change ``--port`` if needed and expose the same port on the client side.

**Docker** — requires `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ (or equivalent) on the host. Pull an image tag from :doc:`../install` or the **Container** row in the selector above; substitute ``<CUOPT_IMAGE>`` below.

Entrypoint mode (recommended when you are not passing an explicit command):

.. code-block:: bash

   docker run --gpus all -it --rm -p 5001:5001 \
     -e CUOPT_SERVER_TYPE=grpc \
     <CUOPT_IMAGE>

Or invoke the binary explicitly:

.. code-block:: bash

   docker run --gpus all -it --rm -p 5001:5001 \
     <CUOPT_IMAGE> \
     cuopt_grpc_server --port 5001 --workers 1

.. note::

   The container image defaults to the Python **REST** server when ``CUOPT_SERVER_TYPE`` is unset and you do not override the command; setting ``CUOPT_SERVER_TYPE=grpc`` selects ``cuopt_grpc_server``. Extra environment variables (``CUOPT_SERVER_PORT``, ``CUOPT_GPU_COUNT``, ``CUOPT_GRPC_ARGS``) and TLS are documented in :doc:`Advanced configuration <advanced>`.

Point the client at the server
==============================

On the machine where you run Python, the C API, or ``cuopt_cli`` (use ``127.0.0.1`` if the server is on the same host):

.. code-block:: bash

   export CUOPT_REMOTE_HOST=<gpu-hostname-or-ip>
   export CUOPT_REMOTE_PORT=5001

Optional TLS and tuning variables are in :doc:`advanced`.

Minimal Python example (LP)
============================

The script is the same for **local** or **remote** solves: with the exports above, the client library forwards to ``cuopt_grpc_server``; without them, the solve runs locally (where a GPU is available).
Please make sure the server is running before running the client.

:download:`remote_lp_demo.py <examples/remote_lp_demo.py>`

.. literalinclude:: examples/remote_lp_demo.py
   :language: python
   :linenos:

Run the script from your NVIDIA cuOpt Python environment. From a **repository checkout** (repo root):

.. code-block:: bash

   python docs/cuopt/source/cuopt-grpc/examples/remote_lp_demo.py

Or, after :download:`downloading <examples/remote_lp_demo.py>` the file into your current directory:

.. code-block:: bash

   python remote_lp_demo.py

You should see an optimal termination. To solve **locally**, unset the remote variables and rerun with the **same** path you used above:

.. code-block:: bash

   unset CUOPT_REMOTE_HOST CUOPT_REMOTE_PORT
   python remote_lp_demo.py

Minimal ``cuopt_cli`` example (LP)
==================================

The same **LP** is available as MPS. With ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT`` set as above, ``cuopt_cli`` forwards the solve to the remote server; unset them for a **local** run (GPU on that machine).
Please make sure the server is running before running the client.

:download:`remote_lp_demo.mps <examples/remote_lp_demo.mps>`

.. literalinclude:: examples/remote_lp_demo.mps
   :language: text

From a **repository checkout** (repo root):

.. code-block:: bash

   cuopt_cli docs/cuopt/source/cuopt-grpc/examples/remote_lp_demo.mps

Or, after :download:`downloading <examples/remote_lp_demo.mps>` the MPS into your current directory:

.. code-block:: bash

   cuopt_cli remote_lp_demo.mps

To solve **locally** with the same file:

.. code-block:: bash

   unset CUOPT_REMOTE_HOST CUOPT_REMOTE_PORT
   cuopt_cli remote_lp_demo.mps

More options (time limits, relaxation): :doc:`../cuopt-cli/quick-start` and :doc:`examples`.

**C API** — With the same environment variables set, call ``solve_lp`` / ``solve_mip`` as in :doc:`../cuopt-c/lp-qp-milp/lp-qp-milp-c-api`.

More patterns (MPS variants, custom gRPC): :doc:`examples`.

Next steps
==========

* :doc:`../install` — Top-level install selector (all interfaces), including **Container** pulls.
* :doc:`advanced` — TLS / mTLS, Docker environment reference, tuning, limitations, troubleshooting.
* :doc:`examples` — Additional client examples and links to LP/MILP sample collections.
* :doc:`api` and :doc:`grpc-server-architecture` — RPC summary and server behavior overview.

See :doc:`../system-requirements` for GPU, CUDA, and OS requirements.
