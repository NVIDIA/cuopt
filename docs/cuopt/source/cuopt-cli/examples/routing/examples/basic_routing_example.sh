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
# Basic Routing CLI Example
#
# This example demonstrates how to use the cuopt_sh CLI tool to solve
# a simple routing problem.
#
# Requirements:
#   - cuOpt server running on localhost:5000
#   - cuopt_sh CLI tool installed
#
# Expected Response:
#   JSON output with optimized routes
#

# Set server connection details
# Update these if your server is running on a different IP/port
export ip="localhost"
export port=5000

# Create sample data file
echo '{"cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
 "task_data": {"task_locations": [0, 1]},
 "fleet_data": {"vehicle_locations": [[0, 0], [0, 0]]}}' > data.json

# Invoke the CLI
# -i: IP address of the cuOpt server
# -p: Port number of the cuOpt server
cuopt_sh data.json -i $ip -p $port

# Clean up
rm -f data.json
