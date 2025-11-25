#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
