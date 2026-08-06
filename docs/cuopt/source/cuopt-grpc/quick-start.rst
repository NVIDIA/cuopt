..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

===========
Quick Start
===========

This page walks through a minimal LP against ``cuopt_grpc_server`` in two
ways:

1. **Remote execution** — set ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT``
   on the client; the same Python, C (``cuOptSolve``), or ``cuopt_cli`` APIs
   you use locally forward to the GPU server with **no code changes**.
2. **Python async gRPC client** — the same LP, with host and port passed to
   ``Client(...)`` for explicit job control (submit / wait / result / delete).

Start the GPU server, then try remote execution first; the gRPC-client
variant follows immediately after that demo. Full client docs:
:doc:`python-async-client`. **Custom** clients call ``CuOptRemoteService``
directly (see :doc:`api`).

.. note::

   **Problem types:** **LP**, **MIP**, and **QP** are supported today.
   **Routing** (VRP, TSP, PDP) over gRPC is **not** available; for remote
   routing, use the HTTP/JSON :doc:`REST self-hosted server <../cuopt-server/index>`.
   This guide is **not** the REST server.

How Remote Execution Works
==========================

1. **GPU server** — On the machine with the GPU, run ``cuopt_grpc_server``
   (bare metal or in the cuOpt container) so it listens on a TCP port
   (default **5001**).
2. **Client machine** — On the machine where you invoke the solver (which may
   be the same host), install the NVIDIA cuOpt client libraries. Set
   ``CUOPT_REMOTE_HOST`` to the **GPU server's** hostname or IP and
   ``CUOPT_REMOTE_PORT`` to the listen port.
3. **Solve** — Call the same APIs you would for a local solve. The integrated
   client opens a gRPC channel, streams the problem, and retrieves the result.
   Unset the two variables to solve **locally** again (local mode still needs
   a GPU on the client machine where applicable).

Install NVIDIA cuOpt
====================

Use the selector below on the **GPU server** and on **client** machines that
need Python, the C API, or ``cuopt_cli``. It is pre-set to **C (libcuopt)**
because that bundle ships ``cuopt_grpc_server``, ``cuopt_cli``, and libraries
together; switch to **Python** if you only need Python packages on a
lightweight client.

.. install-selector::
   :default-iface: c

Verify the server binary on the **GPU server** after installing the C/libcuopt
bundle (that package ships ``cuopt_grpc_server``). A Python-only client install
does not include this binary:

.. code-block:: bash

   cuopt_grpc_server --help

For the same install selector with **Container** / registry choices (Docker Hub or NGC), see :doc:`../install`.

Run the gRPC Server (GPU Server)
================================

**Bare metal** — after activating the same environment you used to install NVIDIA cuOpt:

.. code-block:: bash

   cuopt_grpc_server --port 5001 --workers 1

Leave the process running. Default port **5001**; change ``--port`` if needed and expose the same port to the client.

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

Minimal Python Example
======================

On the **client machine**, point remote execution at the GPU server (use
``127.0.0.1`` if the server is on the same host):

.. code-block:: bash

   export CUOPT_REMOTE_HOST=<gpu-server-hostname-or-ip>
   export CUOPT_REMOTE_PORT=5001

Optional TLS and tuning variables are in :doc:`advanced`. The same exports
apply to the C API and ``cuopt_cli``.

The script below is the same for **local** or **remote** solves: with the
exports above, the integrated client forwards to ``cuopt_grpc_server``;
without them, the solve runs locally (where a GPU is available).
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

.. rubric:: Same LP via the Python Async gRPC Client
   :class: large-rubric

Remote execution needs no code changes. If you want **explicit** job control
instead, leave ``CUOPT_REMOTE_*`` unset and use the Python async gRPC client.
Pass the GPU server's network location in the ``Client`` constructor
(``host`` and ``port``); it does not read ``CUOPT_REMOTE_*``.

Keep the ``DataModel`` setup and ``SolverSettings`` from the listing above, and
replace everything from the ``Solve`` call onward (line 33 in that listing)
with:

.. code-block:: python

   from cuopt.grpc.linear_programming import Client, JobStatus

   # Network location of cuopt_grpc_server (not CUOPT_REMOTE_*).
   client = Client("localhost", 5001)
   job_id = client.submit(dm, settings)
   try:
       status = client.wait(job_id, timeout=120)
       if status != JobStatus.COMPLETED:
           raise RuntimeError(f"unexpected status: {status}")
       # Pass variable names if you want solution.get_vars() keyed by name.
       solution = client.result(job_id, variable_names=["x0", "x1"])
       print("Termination:", solution.get_termination_reason())
       print("Objective:  ", solution.get_primal_objective())
       print("Primal x:   ", solution.get_primal_solution())
   finally:
       client.delete(job_id)

A full walkthrough of log and incumbent streaming is in
:doc:`python-async-client-examples`. Overview and TLS details:
:doc:`python-async-client`.

Minimal ``cuopt_cli`` Example (LP)
==================================

The same **LP** is available as MPS. With ``CUOPT_REMOTE_HOST`` and
``CUOPT_REMOTE_PORT`` set as in the Python example above, ``cuopt_cli``
forwards the solve to the remote server; unset them for a **local** run
(GPU on that machine).
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

**C API** — With the same environment variables set, call ``cuOptSolve`` as in
:doc:`../cuopt-c/convex/convex-c-api`.

More patterns: :doc:`examples`.

Next Steps
==========

* :doc:`../install` — Top-level install selector (all interfaces), including **Container** pulls.
* :doc:`python-async-client` — Python async gRPC client (explicit jobs).
* :doc:`advanced` — TLS / mTLS, Docker environment reference, tuning, limitations, troubleshooting.
* :doc:`examples` — Additional client examples and links to LP/MIP sample collections.
* :doc:`api` and :doc:`grpc-server-architecture` — RPC summary and server behavior overview.

See :doc:`../system-requirements` for GPU, CUDA, and OS requirements.
