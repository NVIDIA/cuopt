#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION &
# SPDX-License-Identifier: Apache-2.0
# AFFILIATES. All rights reserved.
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
#
# Routing Smoke Test Example - Bash Script Version
#
# This script runs the routing smoke test example using Python inline.
# Users can copy and paste this entire script to run the example.

python -c '
import cudf
from cuopt import routing

# Create cost matrix (symmetric distance matrix for 4 locations)
cost_matrix = cudf.DataFrame(
    [[0, 2, 2, 2],
     [2, 0, 2, 2],
     [2, 2, 0, 2],
     [2, 2, 2, 0]],
    dtype="float32"
)

# Task locations (indices into the cost matrix)
# Tasks at locations 1, 2, and 3
task_locations = cudf.Series([1, 2, 3])

# Number of vehicles
n_vehicles = 2

# Create data model
dm = routing.DataModel(
    cost_matrix.shape[0], n_vehicles, len(task_locations)
)
dm.add_cost_matrix(cost_matrix)
dm.add_transit_time_matrix(cost_matrix.copy(deep=True))

# Configure solver settings
ss = routing.SolverSettings()

# Solve the routing problem
sol = routing.Solve(dm, ss)

# Display results
print(sol.get_route())
print("\n\n****************** Display Routes *************************")
sol.display_routes()
'
