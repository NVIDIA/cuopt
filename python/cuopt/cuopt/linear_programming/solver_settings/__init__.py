# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LP/MIP solver settings package; implementation is in the ``solver_settings`` extension.

Public ``solver_params`` is an immutable tuple so callers cannot mutate the extension
module's internal parameter-name list used for validation.
"""

from .solver_settings import (
    PDLPSolverMode,
    SolverMethod,
    SolverSettings,
    get_solver_parameter_names,
    get_solver_setting,
    solver_params as _solver_params_list,
)

solver_params = tuple(_solver_params_list)

__all__ = [
    "PDLPSolverMode",
    "SolverMethod",
    "SolverSettings",
    "get_solver_parameter_names",
    "get_solver_setting",
    "solver_params",
]
