..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

=====================================
Python Async gRPC Client Examples
=====================================

These snippets build on the :doc:`python-async-client` **Connect and Solve**
example. Start ``cuopt_grpc_server`` first, and pass the server host and port
to ``Client`` (not ``CUOPT_REMOTE_*``). Always call ``delete`` when finished,
and pass ``variable_names`` to ``result()`` if you want named ``get_vars()``.

Log Streaming
=============

After ``submit``, stream solver log lines until the job completes:

.. code-block:: python

   from cuopt.grpc.linear_programming import Client, JobStatus

   client = Client("localhost", 5001)
   job_id = client.submit(dm, settings)
   try:
       client.start_log_stream(
           job_id, callback=lambda line, _done: print(line, flush=True)
       )
       if client.wait(job_id, timeout=120) != JobStatus.COMPLETED:
           raise RuntimeError("job did not complete")

       solution = client.result(job_id, variable_names=["x0", "x1"])
       print(solution.get_termination_reason(), solution.get_primal_objective())
   finally:
       try:
           client.join_log_stream(job_id)
       finally:
           client.delete(job_id)

Incumbent Streaming (MIP)
=========================

Register incumbent callbacks the same way as for a local solve: add a
``GetSolutionCallback`` (from ``cuopt.linear_programming.internals``) on
``SolverSettings`` with
:meth:`~cuopt.linear_programming.solver_settings.SolverSettings.set_mip_callback`.
For gRPC, pass that ``settings`` to ``submit``, then call
``start_incumbent_stream`` with the same ``settings`` so those callbacks
receive incumbents while the job runs.

:download:`incumbent_stream_demo.py <examples/incumbent_stream_demo.py>`

.. literalinclude:: examples/incumbent_stream_demo.py
   :language: python
   :linenos:

See Also
========

* :doc:`python-async-client` — overview and Connect and Solve
* :doc:`python-async-client-api` — API reference
* :doc:`quick-start` — remote execution and the same LP via ``Client``
* :doc:`examples` — remote execution examples (``CUOPT_REMOTE_*``)
