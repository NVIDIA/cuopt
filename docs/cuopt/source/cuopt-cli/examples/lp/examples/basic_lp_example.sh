#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Basic LP Example using cuopt_cli
#
# This example demonstrates how to solve a simple LP problem using cuopt_cli.
# cuopt_cli is a standalone command-line tool that solves LP/MILP problems
# from MPS files without requiring a server.
#
# Requirements:
#   - cuopt_cli installed and available in PATH
#

# Create a sample MPS file
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

# Solve using default settings
cuopt_cli sample.mps

# Clean up
rm -f sample.mps
