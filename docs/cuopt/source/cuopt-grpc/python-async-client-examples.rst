..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

=====================================
Python Async gRPC Client Examples
=====================================

These snippets build on the :doc:`python-async-client` **Connect and solve**
example. Start ``cuopt_grpc_server`` first, and pass the server host and port
to ``Client`` (not ``CUOPT_REMOTE_*``). Always call ``delete`` when finished,
and pass ``variable_names`` to ``result()`` if you want named ``get_vars()``.

Log streaming
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

Incumbent streaming (MIP)
=========================

Register incumbent callbacks the same way as for a local solve: add a
``GetSolutionCallback`` (from ``cuopt.linear_programming.internals``) on
``SolverSettings`` with
:meth:`~cuopt.linear_programming.solver_settings.SolverSettings.set_mip_callback`.
For gRPC, pass that ``settings`` to ``submit``, then call
``start_incumbent_stream`` with the same ``settings`` so those callbacks
receive incumbents while the job runs:

1. ``settings.set_mip_callback(...)`` before ``submit``
2. ``client.submit(problem, settings)``
3. ``client.start_incumbent_stream(job_id, settings=settings)``
4. ``join_incumbent_stream`` after the job finishes, then ``result`` / ``delete``

Runnable script:
:download:`incumbent_stream_demo.py <examples/incumbent_stream_demo.py>`.

.. code-block:: python

   from cuopt.grpc.linear_programming import Client, JobStatus
   from cuopt.linear_programming.internals import GetSolutionCallback
   from cuopt.linear_programming.problem import INTEGER, MAXIMIZE, Problem
   from cuopt.linear_programming.solver_settings import SolverSettings

   class IncumbentPrinter(GetSolutionCallback):
       def get_solution(self, solution, solution_cost, solution_bound, user_data):
           print(f"incumbent cost={float(solution_cost[0]):.4f}")

   problem = Problem("demo_mip")
   x = problem.addVariable(vtype=INTEGER, name="x")
   y = problem.addVariable(vtype=INTEGER, name="y")
   problem.addConstraint(2 * x + 4 * y >= 230)
   problem.addConstraint(3 * x + 2 * y <= 190)
   problem.setObjective(5 * x + 3 * y, sense=MAXIMIZE)

   settings = SolverSettings()
   settings.set_mip_callback(IncumbentPrinter(), None)

   client = Client("localhost", 5001)
   job_id = client.submit(problem, settings)
   try:
       # Pass the same settings so GetSolutionCallback instances receive incumbents.
       client.start_incumbent_stream(job_id, settings=settings)
       if client.wait(job_id, timeout=120) != JobStatus.COMPLETED:
           raise RuntimeError("job did not complete")
       client.join_incumbent_stream(job_id)

       names = [v.getVariableName() for v in problem.getVariables()]
       solution = client.result(job_id, variable_names=names)
       print(solution.get_termination_reason(), solution.get_primal_objective())
   finally:
       client.delete(job_id)

See also
========

* :doc:`python-async-client` — overview and Connect and solve
* :doc:`python-async-client-api` — API reference
* :doc:`quick-start` — remote execution and the same LP via ``Client``
* :doc:`examples` — remote execution examples (``CUOPT_REMOTE_*``)
