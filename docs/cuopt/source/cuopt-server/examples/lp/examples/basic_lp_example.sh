#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Basic LP CLI Example
#
# This example demonstrates how to use the cuopt_sh CLI tool to solve
# a simple LP problem.
#
# Requirements:
#   - cuOpt server running on localhost:5000
#   - cuopt_sh CLI tool installed
#
# Expected Response:
#   JSON output with optimal solution

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
    "maximize": false,
    "solver_config": {
        "tolerances": {
            "optimality": 0.0001
        }
    }
 }' > data.json

# Invoke the CLI
# -t LP: Problem type
# -i: IP address
# -p: Port
# -sl: Show logs
cuopt_sh data.json -t LP -i $ip -p $port -sl

# Clean up
rm -f data.json
