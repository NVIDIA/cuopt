#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# TEST_SKIP: requires multiple GPUs
#
# Multi-GPU PDLP Example using cuopt_cli
#
# This example shows how to shard an LP across several GPUs with PDLP.
# Multi-GPU PDLP is selected by --method 1 (PDLP) together with --num-gpus set
# to -1 (all visible GPUs) or to a value greater than 1.
#
# Requirements:
#   - cuopt_cli installed and available in PATH
#   - At least 2 visible GPUs
#   - An LP in MPS format (multi-GPU PDLP only supports LPs, and helps most on
#     large problems; the tiny model below only illustrates the options)
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

echo "=== Example 1: Use all visible GPUs ==="
cuopt_cli --method 1 --num-gpus -1 sample.mps

echo ""
echo "=== Example 2: Use exactly 2 GPUs ==="
cuopt_cli --method 1 --num-gpus 2 sample.mps

echo ""
echo "=== Example 3: Choose how the problem is split across GPUs ==="
# 0 = auto (default), 1 = KaMinPar, 2 = round robin
cuopt_cli --method 1 --num-gpus -1 --multigpu-pdlp-partitioner 2 sample.mps

# Clean up
rm -f sample.mps
