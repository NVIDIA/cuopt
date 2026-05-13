# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from cuopt._build_variant import CUOPT_PYTHON_BUILD_COMPONENT as _CUOPT_SLICE
except ModuleNotFoundError:
    _CUOPT_SLICE = "FULL"

if _CUOPT_SLICE == "LP":
    raise ImportError(
        "cuopt.distance_engine is not available: this installation was built for LP/math only "
        "(e.g. build.sh --math). Use a routing or full build for distance_engine."
    )

from cuopt.distance_engine.waypoint_matrix import WaypointMatrix
