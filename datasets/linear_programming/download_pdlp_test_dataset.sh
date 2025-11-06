#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# 10 easiest Mittleman instances
datasets=(
    "graph40-40"
    "ex10"
    "datt256_lp"
    "woodlands09"
    "savsched1"
    "nug08-3rd"
    "qap15"
    "scpm1"
    "neos3"
    "a2864"
    "ns1687037"
    "square41"
)

for dataset in "${datasets[@]}"; do
    python benchmarks/linear_programming/utils/get_datasets.py -d "$dataset"
done
