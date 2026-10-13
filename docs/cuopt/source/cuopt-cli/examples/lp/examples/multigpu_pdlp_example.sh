#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-GPU PDLP Example using cuopt_cli
#
# Solve a linear program with multi-GPU PDLP.
# --method 1 selects PDLP.
# --num-gpus -1 uses every GPU visible to the process. A value greater than 1
# selects that many GPUs. The problem is sharded when more than one GPU is
# selected.
#
# Requirements:
#   - cuopt_cli installed and available in PATH
#   - An MPS or LP file, such as afiro_original.mps
#   - More than one visible GPU for a sharded solve

echo "=== PDLP on 1 GPU ==="
cuopt_cli --method 1 --num-gpus 1 ./datasets/linear_programming/afiro_original.mps

echo ""
echo "=== mPDLP on all visible GPUs ==="
cuopt_cli --method 1 --num-gpus -1 ./datasets/linear_programming/afiro_original.mps

