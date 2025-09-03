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

from cuopt_server.utils.job_queue import SolverLPJob
from cuopt_server.utils.linear_programming.data_definition import LPData
from cuopt_server.utils.linear_programming.solver import (
    create_data_model as lp_create_data_model,
    create_solver as lp_create_solver,
)
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


def build_lp_datamodel_from_json(data):
    """
    data: A valid dictionary or a json file-path with
          valid format as per open-api spec.
    """

    if isinstance(data, dict):
        data = LPData.parse_obj(data)
    elif os.path.isfile(data):
        with open(data, "r") as f:
            data = json.loads(f.read())
            # Remove this once we support variable names
            data.pop("variable_names")
            data = LPData.parse_obj(data)
    else:
        raise ValueError(
            f"Invalid type : {type(data)} has been provided as input, "
            "requires json input"
        )

    stub_id = 9999
    stub_warnings = []
    job = SolverLPJob(stub_id, data, None, stub_warnings)
    # transform data into digestible format
    job._transform(job.LP_data)
    data = job.get_data()

    _, data_model = lp_create_data_model(data)
    _, solver_settings = lp_create_solver(data, None)

    return data_model, solver_settings
