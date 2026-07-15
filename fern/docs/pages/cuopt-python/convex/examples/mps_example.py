# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reading a Problem from an MPS File
==================================

This example loads an optimization problem from a file with ``Problem.read``
and solves it. ``Problem.read`` dispatches on the file extension:

* ``.mps`` / ``.qps`` (and ``.gz`` / ``.bz2`` variants) use the MPS reader
* ``.lp`` (and compressed variants) use the LP reader.

The bundled ``sample.mps`` is a small LP with optimal objective ``-0.36``.
"""

import os

from cuopt.linear_programming.problem import Problem


def main():
    # Resolve the sample file next to this script so it runs from any directory.
    mps_path = os.path.join(os.path.dirname(__file__), "sample.mps")

    problem = Problem.read(mps_path)
    problem.solve()

    print(f"Status: {problem.Status}")
    print(f"Number of variables: {problem.NumVariables}")
    print(f"Objective value = {problem.ObjValue}")
    for var in problem.getVariables():
        print(f"{var.VariableName} = {var.Value}")


if __name__ == "__main__":
    main()
