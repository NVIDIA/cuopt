# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from cuopt_server.utils.utils import build_routing_datamodel_from_json
from cuopt import routing
from cuopt.linear_programming.solver_settings import SolverSettings
import cuopt_mps_parser
import os
import json
from typing import NamedTuple



# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from cuopt_server.utils.routing.data_definition import OptimizedRoutingData
from cuopt_server.utils.routing.solver import (
    create_data_model as routing_create_data_model,
    create_solver as routing_create_solver,
    prep_optimization_data as routing_prep_optimization_data,
)
from cuopt_server.utils.solver import populate_optimization_data


def build_routing_datamodel_from_json(data):
    """
    data: A valid dictionary or a json file-path with
          valid format as per open-api spec.
    """

    if isinstance(data, dict):
        pass
    elif os.path.isfile(data):
        with open(data, "r") as f:
            data = dict(OptimizedRoutingData.parse_obj(json.loads(f.read())))
    else:
        raise ValueError(
            f"Invalid type : {type(data)} has been provided as input, "
            "requires json input"
        )

    optimization_data = populate_optimization_data(**data)
    (
        optimization_data,
        cost_matrix,
        travel_time_matrix,
        _,
    ) = routing_prep_optimization_data(optimization_data)
    _, data_model = routing_create_data_model(
        optimization_data,
        cost_matrix=cost_matrix,
        travel_time_matrix=travel_time_matrix,
    )

    _, solver_settings = routing_create_solver(optimization_data)

    return data_model, solver_settings


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

    total_objective_value:float = -1
    vehicle_count:int = -1
    cost:float = -1
    prize:float = -1
    travel_time:float = -1
    solver_time:float = -1
    gpu_memory_usage:float = -1
    git_commit: str = ""
    date_time: str = ""

class LPMetrics(NamedTuple):

    primal_objective_value:float = -1
    solver_time:float = -1
    gpu_memory_usage:float = -1
    git_commit: str = ""
    date_time: str = ""


def get_metrics(d_type):
    if d_type == "mip":
        return {
        "primal_objective_value": {
            "threshold": 1,
            "unit": "primal_objective_value"
        },
        "solver_time": {
            "threshold": 1,
            "unit": "seconds"
        },
        "mip_gap": {
            "threshold": 1,
            "unit": "mip_gap"
        },
        "max_constraint_violation": {
            "threshold": 1,
            "unit": "max"
        },
        "max_int_violation": {
            "threshold": 1,
            "unit": "max"
        },
        "max_variable_bound_violation": {
            "threshold": 1,
            "unit": "max"
        }
    }
    elif dtype == "lp":
        return {
        "primal_objective_value": {
            "threshold": 1,
            "unit": "primal_objective_value",
            "bks": {
                "value": -282.9604743,
                "threshold": 1
            }
        },
        "solver_time": {
            "threshold": 1,
            "unit": "seconds"
        },
        "nb_iterations": {
            "threshold": 1,
            "unit": "num_iterations"
        }
    }

def get_configuration(data_file, data_file_path, d_type):

    data = {}
    test_name = None
    requested_metrics = {}

    if d_type == "lp" or d_type == "mip":
        with open(data_file_path+"/"+d_type+"_config.json") as f:
            data = json.load(f)
        test_name = data_file.split('/')[-1].split('.')[0]
        data_model, solver_settings = build_datamodel_from_mps(data_file)
        requested_metrics = data["metrics"]
    else:
        with open(data_file) as f:
            data = json.load(f)
        test_name = data["test_name"]
        data_model, solver_settings = build_routing_datamodel_from_json(data_file_path+"/"+data["file_name"])
        requested_metrics = data["metrics"]

    return test_name, data_model, solver_settings, requested_metrics
