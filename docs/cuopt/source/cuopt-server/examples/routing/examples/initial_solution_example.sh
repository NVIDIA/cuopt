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
# Initial Solution Routing CLI Example
#
# This example demonstrates how to use a previous solution as an initial
# solution for a new request using the cuopt_sh CLI tool.
#
# Requirements:
#   - cuOpt server running on localhost:5000
#   - cuopt_sh CLI tool installed
#   - jq installed for JSON parsing
#
# Features:
#   - Save previous solution with -k flag
#   - Use previous reqId as initial solution with -id flag
#   - Delete saved solutions with -d flag
#

# Set server connection details
export ip="localhost"
export port=5000

# Create sample data file
echo '{"cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
 "task_data": {"task_locations": [0, 1]},
 "fleet_data": {"vehicle_locations": [[0, 0], [0, 0]]}}' > data.json

echo "=== Step 1: Solve and save solution ==="
# Solve and keep the solution (-k flag)
# Extract the reqId using jq
reqId=$(cuopt_sh data.json -i $ip -p $port -k | sed "s/'/\"/g" | jq -r '.reqId')

echo "Saved solution with reqId: $reqId"

echo ""
echo "=== Step 2: Use saved solution as initial solution ==="
# Use the previous reqId as initial solution
cuopt_sh data.json -i $ip -p $port -id $reqId

echo ""
echo "=== Step 3: Clean up saved solutions ==="
# Delete the saved solution to free memory
cuopt_sh -i $ip -p $port -d $reqId

echo "Solution deleted"

# Clean up data file
rm -f data.json
