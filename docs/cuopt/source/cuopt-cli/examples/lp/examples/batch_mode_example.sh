#!/bin/bash
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

# Set server connection details
export ip="localhost"
export port=5000

# Create sample MPS file
echo "* optimize
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
ENDATA" > sample.mps

echo "=== Solving Multiple MPS Files in Batch Mode ==="
# Submit multiple MPS files at once
# -t LP: Problem type
# -ss: Solver settings (JSON format)
cuopt_sh sample.mps sample.mps sample.mps -t LP -i $ip -p $port -ss '{"tolerances": {"optimality": 0.0001}, "time_limit": 5}'

# Clean up
rm -f sample.mps

echo ""
echo "Note: Batch mode is only available for LP with MPS files, not for MILP."

