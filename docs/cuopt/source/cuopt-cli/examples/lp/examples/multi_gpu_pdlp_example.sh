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
#   - An LP in LP or MPS format (multi-GPU PDLP only supports LPs, and helps
#     most on large problems; the tiny model below only illustrates the options)
#

# Create a sample LP file
cat > sample.lp << 'LP'
Minimize
  -0.2 VAR1 + 0.1 VAR2
Subject To
  ROW1: 3 VAR1 + 4 VAR2 <= 5.4
  ROW2: 2.7 VAR1 + 10.1 VAR2 <= 4.9
End
LP

echo "=== Example 1: Use all visible GPUs ==="
cuopt_cli --method 1 --num-gpus -1 sample.lp

echo ""
echo "=== Example 2: Use exactly 2 GPUs ==="
cuopt_cli --method 1 --num-gpus 2 sample.lp

echo ""
echo "=== Example 3: Choose how the problem is split across GPUs ==="
# 0 = auto (default), 1 = KaMinPar, 2 = round robin
cuopt_cli --method 1 --num-gpus -1 --multigpu-pdlp-partitioner 2 sample.lp

# Clean up
rm -f sample.lp
