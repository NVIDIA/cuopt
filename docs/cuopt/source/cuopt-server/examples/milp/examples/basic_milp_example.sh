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
# Basic MILP CLI Example
#
# This example demonstrates how to use the cuopt_sh CLI tool to solve
# a simple MILP (Mixed Integer Linear Programming) problem.
#
# The main difference from LP is the variable_types field specifying
# integer variables.
#
# Requirements:
#   - cuOpt server running on localhost:5000
#   - cuopt_sh CLI tool installed
#

# Set server connection details
export ip="localhost"
export port=5000

# Create MILP data file
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
    "variable_names": ["x", "y"],
    "variable_types": ["I", "I"],
    "maximize": "False",
    "solver_config": {
        "time_limit": 30
    }
 }' > data.json

# Invoke the CLI
# -t LP: Problem type (same for MILP)
# -i: IP address
# -p: Port
# -sl: Show logs
# -il: Show incumbent logs
cuopt_sh data.json -t LP -i $ip -p $port -sl -il

# Clean up
rm -f data.json
