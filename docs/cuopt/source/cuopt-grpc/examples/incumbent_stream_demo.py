# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MIP incumbent streaming via the Python async gRPC client.

Same callback registration as a local solve
(:meth:`SolverSettings.set_mip_callback`), plus
:meth:`Client.start_incumbent_stream` so those callbacks fire while the
remote job runs.

Start the server first::

    cuopt_grpc_server --port 50051 --workers 1

Then::

    python incumbent_stream_demo.py --host localhost --port 50051
"""

from __future__ import annotations

import argparse
import sys

from cuopt.grpc.linear_programming import Client, JobStatus
from cuopt.linear_programming.internals import GetSolutionCallback
from cuopt.linear_programming.problem import INTEGER, MAXIMIZE, Problem
from cuopt.linear_programming.solver_settings import SolverSettings


class IncumbentPrinter(GetSolutionCallback):
    """Same callback type used for local ``problem.solve(settings)``."""

    def __init__(self):
        super().__init__()
        self.entries = []

    def get_solution(self, solution, solution_cost, solution_bound, user_data):
        cost = float(solution_cost[0])
        values = solution.tolist()
        self.entries.append({"cost": cost, "solution": values})
        print(f"incumbent cost={cost:.4f} values={values}", flush=True)


def build_problem():
    problem = Problem("incumbent_stream_demo")
    x = problem.addVariable(lb=0, ub=10, vtype=INTEGER, name="x")
    y = problem.addVariable(lb=0, ub=10, vtype=INTEGER, name="y")
    problem.addConstraint(x + y <= 10, name="c1")
    problem.addConstraint(x - y >= 0, name="c2")
    problem.setObjective(x + 2 * y, sense=MAXIMIZE)
    return problem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args(argv)

    printer = IncumbentPrinter()
    settings = SolverSettings()
    # Local solves: register callback, then problem.solve(settings).
    # Remote: same registration — submit enables server-side tracking, and
    # start_incumbent_stream(..., settings=settings) delivers into printer.
    settings.set_mip_callback(printer, None)
    settings.set_parameter("time_limit", 30)

    problem = build_problem()
    client = Client(args.host, args.port)
    job_id = client.submit(problem, settings)
    client.start_incumbent_stream(job_id, settings=settings)

    status = client.wait(job_id, timeout=120)
    if status != JobStatus.COMPLETED:
        print(f"unexpected status: {status}", file=sys.stderr)
        return 1
    client.join_incumbent_stream(job_id)

    names = [v.getVariableName() for v in problem.getVariables()]
    solution = client.result(job_id, variable_names=names)
    print(
        f"final reason={solution.get_termination_reason()} "
        f"obj={solution.get_primal_objective()}"
    )
    client.delete(job_id)

    if not printer.entries:
        print(
            "no incumbents delivered via settings callbacks", file=sys.stderr
        )
        return 1
    print(f"OK ({len(printer.entries)} incumbent(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
