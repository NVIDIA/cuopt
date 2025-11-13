#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Basic MILP Example using cuopt_cli
#
# This example demonstrates how to solve a Mixed Integer Programming (MIP)
# problem using cuopt_cli. The main difference from LP is that variables
# are marked as integers in the MPS file.
#
# Requirements:
#   - cuopt_cli installed and available in PATH
#

# Create MILP problem MPS file
echo "* Optimal solution -28
NAME          MIP_SAMPLE
ROWS
 N  OBJ
 L  C1
 L  C2
 L  C3
COLUMNS
 MARK0001  'MARKER'                 'INTORG'
   X1        OBJ             -7
   X1        C1              -1
   X1        C2               5
   X1        C3              -2
   X2        OBJ             -2
   X2        C1               2
   X2        C2               1
   X2        C3              -2
 MARK0001  'MARKER'                 'INTEND'
RHS
   RHS       C1               4
   RHS       C2              20
   RHS       C3              -7
BOUNDS
 UP BOUND     X1               10
 UP BOUND     X2               10
ENDATA" > mip_sample.mps

# Solve the MIP problem with custom parameters
cuopt_cli --mip-absolute-gap 0.01 --time-limit 10 mip_sample.mps

# Clean up
rm -f mip_sample.mps
