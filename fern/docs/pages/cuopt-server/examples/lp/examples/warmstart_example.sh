#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

set -e  # Exit on any command failure

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
        },
        "pdlp_solver_mode": 1
    }
 }' > data.json

echo "=== Step 1: Solve and save solution for warmstart ==="
# Solve and keep the solution (-k flag)
output=$(cuopt_sh -t LP data.json -i $ip -p $port -k)
if [ $? -ne 0 ]; then
    echo "Error: Failed to solve LP problem"
    rm -f data.json
    exit 1
fi

reqId=$(echo "$output" | sed "s/'/\"/g" | sed 's/False/false/g' | jq -r '.reqId')
if [ -z "$reqId" ] || [ "$reqId" = "null" ]; then
    echo "Error: Failed to extract reqId from response"
    rm -f data.json
    exit 1
fi

echo "Saved solution with reqId: $reqId"

echo ""
echo "=== Step 2: Use saved solution as warmstart ==="
# Use the previous reqId as warmstart (-wid flag)
cuopt_sh data.json -t LP -i $ip -p $port -wid $reqId

# Clean up
rm -f data.json

echo ""
echo "Note: Warmstart is only supported for LP problems, not MILP."
