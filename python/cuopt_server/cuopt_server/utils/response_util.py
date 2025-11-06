# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
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

from typing import Dict

from .routing.optimization_data_model import OptimizationDataModel


def get_full_response(
    end_point_response: Dict,
    optimization_data: OptimizationDataModel,
    status: int,
    return_status: bool,
    return_data_state: bool,
    warnings=[],
    notes=[],
    reqId="",
):
    full_response = {}
    full_response["response"] = end_point_response

    if reqId:
        full_response["reqId"] = reqId

    if warnings:
        full_response["warnings"] = warnings

    if notes:
        full_response["notes"] = notes

    if return_status:
        full_response["solver_status"] = status

    if return_data_state:
        full_response["optimization_data"] = (
            optimization_data.get_optimization_data()
        )

    return full_response
