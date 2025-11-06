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
# LP Batch Mode CLI Example
#
# This example demonstrates how to solve multiple LP problems in batch mode
# using MPS files with the cuopt_sh CLI tool.
#
# Note: Batch mode works only with MPS files in CLI and is not available for MILP.
#
# Requirements:
#   - cuOpt server running on localhost:5000
#   - cuopt_sh CLI tool installed
#
set -e

# Set server connection details
export ip="localhost"
export port=5000

# Create temporary MPS file and ensure cleanup on exit
mps_file=$(mktemp --suffix=.mps)
trap "rm -f \"$mps_file\"" EXIT

# Create sample MPS file
cat > "$mps_file" << 'EOF'
* optimize
*  cost = -0.2 * VAR1 + 0.1 * VAR2
* subject to
*  3 * VAR1 + 4 * VAR2 <= 5.4
*  2.7 * VAR1 + 10.1 * VAR2 <= 4.9
NAME   good-1
ROWS
 N  COST
 L  ROW1
 L  ROW2
COLUMNS
   VAR1      COST      -0.2
   VAR1      ROW1      3              ROW2      2.7
   VAR2      COST      0.1
   VAR2      ROW1      4              ROW2      10.1
RHS
   RHS1      ROW1      5.4            ROW2      4.9
ENDATA
EOF

echo "=== Solving Multiple MPS Files in Batch Mode ==="
# Submit multiple MPS files at once
# -t LP: Problem type
# -ss: Solver settings (JSON format)
cuopt_sh "$mps_file" "$mps_file" "$mps_file" -t LP -i $ip -p $port -ss '{"tolerances": {"optimality": 0.0001}, "time_limit": 5}'

echo ""
echo "Note: Batch mode is only available for LP with MPS files, not for MILP."
