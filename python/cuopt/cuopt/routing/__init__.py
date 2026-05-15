# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from cuopt._build_variant import CUOPT_PYTHON_BUILD_COMPONENT as _CUOPT_SLICE
except ModuleNotFoundError:
    _CUOPT_SLICE = "FULL"

if _CUOPT_SLICE == "LP":
    raise ImportError(
        "cuopt.routing is not available: this installation was built for LP/math only "
        "(e.g. build.sh --math). Use a routing or full build for vehicle routing."
    )

from cuopt.routing.assignment import Assignment, SolutionStatus
from cuopt.routing.utils import (
    add_vehicle_constraints,
    create_pickup_delivery_data,
    generate_dataset,
    update_routes_and_vehicles,
)
from cuopt.routing.utils_wrapper import DatasetDistribution
from cuopt.routing.vehicle_routing import BatchSolve, DataModel, Solve, SolverSettings
from cuopt.routing.vehicle_routing_wrapper import ErrorStatus, Objective
