# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MIP incumbent streaming via the Python async gRPC client.

Same ``set_mip_callback`` registration as a local solve, plus
``start_incumbent_stream`` so those callbacks fire while the remote job runs.

Start the server first::

    cuopt_grpc_server --port 5001 --workers 1

Then::

    python incumbent_stream_demo.py
"""

from cuopt.grpc.linear_programming import Client, JobStatus
from cuopt.linear_programming.internals import GetSolutionCallback
from cuopt.linear_programming.problem import INTEGER, MAXIMIZE, Problem
from cuopt.linear_programming.solver_settings import SolverSettings


class IncumbentPrinter(GetSolutionCallback):
    def get_solution(self, solution, solution_cost, solution_bound, user_data):
        print(
            f"incumbent cost={float(solution_cost[0]):.4f} "
            f"values={solution.tolist()}",
            flush=True,
        )


problem = Problem("incumbent_stream_demo")
x = problem.addVariable(lb=0, ub=10, vtype=INTEGER, name="x")
y = problem.addVariable(lb=0, ub=10, vtype=INTEGER, name="y")
problem.addConstraint(x + y <= 10, name="c1")
problem.addConstraint(x - y >= 0, name="c2")
problem.setObjective(x + 2 * y, sense=MAXIMIZE)

settings = SolverSettings()
settings.set_mip_callback(IncumbentPrinter(), None)
settings.set_parameter("time_limit", 30)

client = Client("localhost", 5001)
job_id = client.submit(problem, settings)
try:
    client.start_incumbent_stream(job_id, settings=settings)
    if client.wait(job_id, timeout=120) != JobStatus.COMPLETED:
        raise RuntimeError("job did not complete")
    client.join_incumbent_stream(job_id)
    names = [v.getVariableName() for v in problem.getVariables()]
    solution = client.result(job_id, variable_names=names)
    print(solution.get_termination_reason(), solution.get_primal_objective())
finally:
    client.delete(job_id)
