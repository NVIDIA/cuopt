#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# LP Warmstart CLI Example
#
# This example demonstrates how to use a previous solution as warmstart
# for a new LP request using the cuopt_sh CLI tool.
#
# Note: Warmstart is only applicable to LP, not for MILP.
#
# Requirements:
#   - cuOpt server running on localhost:5000
#   - cuopt_sh CLI tool installed
#   - jq installed for JSON parsing
#

# Set server connection details
export ip="localhost"
export port=5000

# Create LP data file
echo '{
    "csr_constraint_matrix": {
        "offsets": [0, 2, 4],
        "indices": [0, 1, 0, 1],
        "values": [3.0, 4.0, 2.7, 10.1]
    },
    "constraint_bounds": {
        "upper_bounds": [5.4, 4.9],
        "lower_bounds": ["ninf", "ninf"]
    },
    "objective_data": {
        "coefficients": [0.2, 0.1],
        "scalability_factor": 1.0,
        "offset": 0.0
    },
    "variable_bounds": {
        "upper_bounds": ["inf", "inf"],
        "lower_bounds": [0.0, 0.0]
    },
    "maximize": "False",
    "solver_config": {
        "tolerances": {
            "optimality": 0.0001
        }
    }
 }' > data.json

echo "=== Step 1: Solve and save solution for warmstart ==="
# Solve and keep the solution (-k flag)
reqId=$(cuopt_sh -t LP data.json -i $ip -p $port -k | sed "s/'/\"/g" | sed 's/False/false/g' | jq -r '.reqId')

echo "Saved solution with reqId: $reqId"

echo ""
echo "=== Step 2: Use saved solution as warmstart ==="
# Use the previous reqId as warmstart (-wid flag)
cuopt_sh data.json -t LP -i $ip -p $port -wid $reqId

# Clean up
rm -f data.json

echo ""
echo "Note: Warmstart is only supported for LP problems, not MILP."
