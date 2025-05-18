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

from cuopt.routing.assignment import Assignment, SolutionStatus
from cuopt.routing.utils import (
    add_vehicle_constraints,
    create_pickup_delivery_data,
    generate_dataset,
    update_routes_and_vehicles,
)
from cuopt.routing.utils_wrapper import DatasetDistribution
from cuopt.routing.vehicle_routing import DataModel, Solve, SolverSettings
from cuopt.routing.vehicle_routing_wrapper import ErrorStatus, Objective
