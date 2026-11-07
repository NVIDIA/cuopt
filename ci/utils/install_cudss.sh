#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Clean metadata & install cudss
# Pin to 1.0.x to match the runtime cuDSS pinned in dependencies.yaml.
if command -v dnf &> /dev/null; then
    # The cudss-cuda-<major> metapackage pulls in the runtime, devel and static packages (the static
    # library is needed by CMake) and registers the default-path links that 1.0's library packages no longer do.
    if [ "$(echo "$CUDA_VERSION" | cut -d. -f1)" -ge 13 ] && [ "$(echo "$CUDA_VERSION" | cut -d. -f1)" -lt 14 ]; then
        dnf -y install "cudss-cuda-13-1.0.*"
    else
        dnf -y install "cudss-cuda-12-1.0.*"
    fi
else
    echo "dnf not found. Cannot install cudss dependencies."
    exit 1
fi
