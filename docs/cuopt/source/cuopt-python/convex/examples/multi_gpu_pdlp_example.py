# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# TEST_SKIP: requires multiple GPUs

"""
Multi-GPU PDLP Example

This example demonstrates:
- Selecting the PDLP method
- Using all visible GPUs (num_gpus = -1) or a fixed number of GPUs
- Choosing how the problem is partitioned across GPUs

Multi-GPU PDLP shards the problem across GPUs. It helps most on large LPs,
or LPs that do not fit in the memory of a single GPU. The small model below
only illustrates the settings. Multi-GPU PDLP supports LPs only, and needs at
least 2 visible GPUs.

Problem:
    Maximize: 2*x + y
    Subject to:
        4*x + 10*y <= 130
        8*x - 3*y >= 40
        x, y >= 0
"""

from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_METHOD,
    CUOPT_NUM_GPUS,
    CUOPT_MULTIGPU_PDLP_PARTITIONER,
)
from cuopt.linear_programming.solver_settings import (
    SolverSettings,
    SolverMethod,
)


def main():
    """Run the multi-GPU PDLP example."""
    problem = Problem("Simple LP")

    x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
    y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

    problem.addConstraint(4 * x + 10 * y <= 130, name="c1")
    problem.addConstraint(8 * x - 3 * y >= 40, name="c2")

    problem.setObjective(2 * x + y, sense=MAXIMIZE)

    settings = SolverSettings()
    # Multi-GPU PDLP needs the PDLP method
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
    # -1 uses all GPUs visible to the process; a value > 1 uses that many
    settings.set_parameter(CUOPT_NUM_GPUS, -1)
    # How to split the problem: 0 = auto (default), 1 = KaMinPar,
    # 2 = round robin
    settings.set_parameter(CUOPT_MULTIGPU_PDLP_PARTITIONER, 0)

    problem.solve(settings)

    print(f"Status: {problem.Status.name}")
    print(f"Solved in {problem.SolveTime:.4f} seconds")
    print(f"x = {x.getValue()}, y = {y.getValue()}")
    print(f"Objective value = {problem.ObjValue}")


if __name__ == "__main__":
    main()
