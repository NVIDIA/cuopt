#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Solver Parameters Example using cuopt_cli
#
# This example demonstrates how to customize solver behavior using various
# command line parameters with cuopt_cli.
#
# Requirements:
#   - cuopt_cli installed and available in PATH
#

# Create a sample MPS file for testing different parameters
echo "* optimize
*  cost = -0.2 * VAR1 + 0.1 * VAR2
* subject to
*  3 * VAR1 + 4 * VAR2 <= 5.4
*  2.7 * VAR1 + 10.1 * VAR2 <= 4.9
NAME          SAMPLE
ROWS
 N  COST
 L  ROW1
 L  ROW2
COLUMNS
 VAR1      COST                -0.2
 VAR1      ROW1                3.0
 VAR1      ROW2                2.7
 VAR2      COST                0.1
 VAR2      ROW1                4.0
 VAR2      ROW2               10.1
RHS
 RHS1      ROW1                5.4
 RHS1      ROW2                4.9
ENDATA" > sample.mps

echo "=== Example 1: Set absolute primal tolerance and PDLP solver mode ==="
cuopt_cli --absolute-primal-tolerance 0.0001 --pdlp-solver-mode 1 sample.mps

echo ""
echo "=== Example 2: Set time limit and use specific solver method (PDLP only) ==="
cuopt_cli --time-limit 5 --method 1 sample.mps

echo ""
echo "=== Example 3: Redirect output to log file and solution file ==="
cuopt_cli --log-to-console false --log-file sample.log --solution-file sample.sol sample.mps
echo "Log and solution files created: sample.log, sample.sol"

# Clean up
rm -f sample.mps sample.log sample.sol
