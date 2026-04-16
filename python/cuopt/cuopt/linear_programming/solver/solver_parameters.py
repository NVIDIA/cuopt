# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible module; LP parameter helpers live in solver_settings."""

from cuopt.linear_programming.solver_settings.solver_settings import (
    get_solver_parameter_names,
    get_solver_setting,
    solver_params,
)

import cuopt.linear_programming.solver_settings.solver_settings as _solver_settings_ext

for _name in dir(_solver_settings_ext):
    if _name.startswith("CUOPT_"):
        globals()[_name] = getattr(_solver_settings_ext, _name)
