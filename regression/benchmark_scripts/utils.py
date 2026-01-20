# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from cuopt_server.utils.utils import build_routing_datamodel_from_json
from cuopt.linear_programming.solver_settings import SolverSettings
import cuopt_mps_parser
import os
import json
from typing import NamedTuple


def build_datamodel_from_mps(data):
    """
    data: A file in mps format
    """

    if os.path.isfile(data):
        data_model = cuopt_mps_parser.ParseMps(data)
    else:
        raise ValueError(
            f"Invalid type : {type(data)} has been provided as input, "
            "requires mps input"
        )
    solver_settings = SolverSettings()

    return data_model, solver_settings


class RoutingMetrics(NamedTuple):
    total_objective_value: float = -1
    vehicle_count: int = -1
    cost: float = -1
    prize: float = -1
    travel_time: float = -1
    solver_time: float = -1
    gpu_memory_usage: float = -1
    git_commit: str = ""
    date_time: str = ""


class LPMetrics(NamedTuple):
    primal_objective_value: float = -1
    solver_time: float = -1
    gpu_memory_usage: float = -1
    git_commit: str = ""
    date_time: str = ""


def get_configuration(data_file, data_file_path, d_type):
    data = {}
    test_name = None
    requested_metrics = {}

    if d_type == "lp" or d_type == "mip":
        with open(data_file_path + "/" + d_type + "_config.json") as f:
            data = json.load(f)
        test_name = data_file.split("/")[-1].split(".")[0]
        data_model, solver_settings = build_datamodel_from_mps(data_file)
        requested_metrics = data["metrics"]
    else:
        with open(data_file) as f:
            data = json.load(f)
        test_name = data["test_name"]
        data_model, solver_settings = build_routing_datamodel_from_json(
            data_file_path + "/" + data["file_name"]
        )
        requested_metrics = data["metrics"]

    return test_name, data_model, solver_settings, requested_metrics
