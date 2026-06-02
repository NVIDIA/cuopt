#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Clean metadata & install cudss
# Pin to 0.7.x to match the runtime cuDSS (pip/conda only ship 0.7); bump once
# nvidia-cudss-cuXX 0.8 is on PyPI and libcudss 0.8 is on conda-forge.
if command -v dnf &> /dev/null; then
    # Adding static library just to please CMAKE requirements
    if [ "$(echo "$CUDA_VERSION" | cut -d. -f1)" -ge 13 ] && [ "$(echo "$CUDA_VERSION" | cut -d. -f1)" -lt 14 ]; then
        dnf -y install "libcudss0-static-cuda-13-0.7.*" "libcudss0-devel-cuda-13-0.7.*" "libcudss0-cuda-13-0.7.*"
    else
        dnf -y install "libcudss0-static-cuda-12-0.7.*" "libcudss0-devel-cuda-12-0.7.*" "libcudss0-cuda-12-0.7.*"
    fi
elif command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y "libcudss-devel=0.7.*"
else
    echo "Neither dnf nor apt-get found. Cannot install cudss dependencies."
    exit 1
fi
