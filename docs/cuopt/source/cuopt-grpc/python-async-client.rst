..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

=============================
Python Async gRPC Client
=============================

The **Python async gRPC client** (``cuopt.grpc.linear_programming.Client``)
is an explicit gRPC client for ``cuopt_grpc_server``. It uses a job lifecycle:
**submit** → **wait** / **status** → **result** → **delete**.

"Async" here means the **job-based** API (submit-and-wait / submit-and-poll). It is
**not** Python ``asyncio``.

For **remote execution** (zero code change via ``CUOPT_REMOTE_HOST`` /
``CUOPT_REMOTE_PORT``), see :doc:`quick-start` and the section overview in
:doc:`index`. Prefer this client when you need cancel, live log streaming,
MIP incumbent streaming, or to manage multiple jobs yourself.

Prerequisites
=============

A running ``cuopt_grpc_server`` on a GPU host (see :doc:`quick-start`):

.. code-block:: bash

   cuopt_grpc_server --port 5001 --workers 1

Connect and Solve
=================

The LP below matches :download:`remote_lp_demo.py <examples/remote_lp_demo.py>`
from the quick start (same constraint matrix and objective).

.. code-block:: python

   import numpy as np
   from cuopt import linear_programming
   from cuopt.grpc.linear_programming import Client, JobStatus

   dm = linear_programming.DataModel()
   dm.set_csr_constraint_matrix(
       np.array([3.0, 4.0, 2.7, 10.1], dtype=np.float64),
       np.array([0, 1, 0, 1], dtype=np.int32),
       np.array([0, 2, 4], dtype=np.int32),
   )
   dm.set_constraint_bounds(np.array([5.4, 4.9], dtype=np.float64))
   dm.set_objective_coefficients(np.array([0.2, 0.1], dtype=np.float64))
   dm.set_maximize(True)
   dm.set_row_types(np.array(["L", "L"]))
   dm.set_variable_lower_bounds(np.array([0.0, 0.0], dtype=np.float64))
   dm.set_variable_upper_bounds(np.array([2.0, np.inf], dtype=np.float64))

   settings = linear_programming.SolverSettings()
   client = Client("localhost", 5001)  # tls=None uses CUOPT_TLS_* if set
   job_id = client.submit(dm, settings)
   try:
       if client.wait(job_id, timeout=120) != JobStatus.COMPLETED:
           raise RuntimeError("job did not complete")
       # Pass names if you want solution.get_vars() keyed by name.
       solution = client.result(job_id, variable_names=["x0", "x1"])
       print(solution.get_termination_reason(), solution.get_primal_objective())
   finally:
       client.delete(job_id)

``Client.submit()`` accepts either a
:class:`~cuopt.linear_programming.data_model.DataModel` or a
:class:`~cuopt.linear_programming.problem.Problem`. Always call
``delete`` after you are done with the job so the server can release state.

Variable Names
==============

``result(job_id, variable_names=...)`` builds the solution object. Pass a list
of variable names (same order as the model columns) if you want
``solution.get_vars()`` keyed by those names. You can omit names and still use
``get_primal_solution()`` and other numeric accessors.

TLS / mTLS
==========

* ``tls=None`` (default) — honor ``CUOPT_TLS_*`` environment variables.
* ``tls=False`` — plain TCP; ignore ``CUOPT_TLS_*``.
* ``tls=TlsConfig(...)`` — explicit PEM text or file paths.

See :doc:`advanced` for server-side TLS and which environment variables apply
to remote execution versus this gRPC client.

Next Steps
==========

* :doc:`python-async-client-examples` — log streaming and incumbent streaming
* :doc:`python-async-client-api` — API reference
* :doc:`quick-start` — start ``cuopt_grpc_server`` and remote execution
* :doc:`api` — low-level ``CuOptRemoteService`` RPCs
