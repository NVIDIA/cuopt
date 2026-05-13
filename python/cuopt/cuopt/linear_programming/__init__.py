# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from cuopt._build_variant import CUOPT_PYTHON_BUILD_COMPONENT as _CUOPT_SLICE
except ModuleNotFoundError:
    _CUOPT_SLICE = "FULL"

if _CUOPT_SLICE == "ROUTING":
    raise ImportError(
        "cuopt.linear_programming is not available: this installation was built for ROUTING only "
        "(e.g. build.sh --routing). Use a math/full build for LP/MILP/QP."
    )

from cuopt.linear_programming import internals
from cuopt.linear_programming.data_model import DataModel
from cuopt.linear_programming.problem import Problem
from cuopt.linear_programming.solution import Solution
from cuopt.linear_programming.solver import BatchSolve, Solve
from cuopt.linear_programming.solver_settings import (
    PDLPSolverMode,
    SolverMethod,
    SolverSettings,
)
