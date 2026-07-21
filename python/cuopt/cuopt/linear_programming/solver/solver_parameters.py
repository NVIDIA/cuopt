# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible module; LP parameter helpers live in ``solver_settings``."""

from cuopt.linear_programming.solver_settings.solver_settings import (
    get_solver_parameter_names,
    get_solver_setting,
    solver_params,
)

import cuopt.linear_programming.solver_settings.solver_settings as _solver_settings_ext

_cuopt_constant_names = tuple(
    f"CUOPT_{p.upper()}" for p in _solver_settings_ext.solver_params
)
for _name in _cuopt_constant_names:
    globals()[_name] = getattr(_solver_settings_ext, _name)

__all__ = (
    "get_solver_parameter_names",
    "get_solver_setting",
    "solver_params",
    *_cuopt_constant_names,
)
