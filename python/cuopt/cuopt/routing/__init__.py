# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dataset-generation and batch helpers (generate_dataset, DatasetDistribution,
# add_vehicle_constraints, ...) are local-testing utilities that require a GPU
# to run. They deliberately live in cuopt.routing.utils and are NOT re-exported
# here, so importing cuopt.routing (e.g. to build/serialize a problem for a
# remote solver) does not pull in that GPU machinery. Import them from
# ``cuopt.routing.utils`` directly when needed.
from cuopt.routing.assignment import Assignment, SolutionStatus
from cuopt.routing.vehicle_routing import (
    BatchSolve,
    DataModel,
    Solve,
    SolverSettings,
)
from cuopt.routing.vehicle_routing_wrapper import ErrorStatus, Objective

__all__ = [
    "Assignment",
    "BatchSolve",
    "DataModel",
    "ErrorStatus",
    "Objective",
    "SolutionStatus",
    "Solve",
    "SolverSettings",
]
