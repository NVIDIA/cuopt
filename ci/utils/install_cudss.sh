#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# cuDSS 1.0 installs under versioned directories (/usr/lib64/libcudss/<CUDA major>/,
# /usr/include/libcudss/<CUDA major>/) but, unlike 0.x, its packages no longer register
# update-alternatives links into the default search paths. Without them, find_package(cudss),
# -lcudss and #include <cudss.h> no longer find it, so register the same links 0.x did.
register_cudss_alternatives() {
    local cuda_major="$1"
    local lib="/usr/lib64/libcudss/${cuda_major}"
    local inc="/usr/include/libcudss/${cuda_major}"
    mkdir -p /usr/lib64/cmake
    update-alternatives --install /usr/lib64/libcudss.so.1 cudss "${lib}/libcudss.so.1" "${cuda_major}0" \
        --slave /usr/lib64/libcudss.so libcudss.so "${lib}/libcudss.so" \
        --slave /usr/lib64/libcudss_mtlayer_gomp.so.1 libcudss_mtlayer_gomp.so.1 "${lib}/libcudss_mtlayer_gomp.so.1" \
        --slave /usr/lib64/libcudss_mtlayer_gomp.so libcudss_mtlayer_gomp.so "${lib}/libcudss_mtlayer_gomp.so" \
        --slave /usr/lib64/libcudss_commlayer_nccl.so libcudss_commlayer_nccl.so "${lib}/libcudss_commlayer_nccl.so" \
        --slave /usr/lib64/libcudss_commlayer_openmpi.so libcudss_commlayer_openmpi.so "${lib}/libcudss_commlayer_openmpi.so" \
        --slave /usr/lib64/libcudss_static.a libcudss_static.a "${lib}/libcudss_static.a" \
        --slave /usr/lib64/cmake/cudss cudss_cmake "${lib}/cmake/cudss" \
        --slave /usr/include/cudss.h cudss.h "${inc}/cudss.h" \
        --slave /usr/include/cudss_data_types.h cudss_data_types.h "${inc}/cudss_data_types.h" \
        --slave /usr/include/cudss_threading_interface.h cudss_threading_interface.h "${inc}/cudss_threading_interface.h" \
        --slave /usr/include/cudss_distributed_interface.h cudss_distributed_interface.h "${inc}/cudss_distributed_interface.h"
    ldconfig
}

# Clean metadata & install cudss
# Pin to 1.0.x to match the runtime cuDSS pinned in dependencies.yaml.
if command -v dnf &> /dev/null; then
    # Adding static library just to please CMAKE requirements
    if [ "$(echo "$CUDA_VERSION" | cut -d. -f1)" -ge 13 ] && [ "$(echo "$CUDA_VERSION" | cut -d. -f1)" -lt 14 ]; then
        dnf -y install "libcudss1-static-cuda-13-1.0.*" "libcudss1-devel-cuda-13-1.0.*" "libcudss1-cuda-13-1.0.*"
        register_cudss_alternatives 13
    else
        dnf -y install "libcudss1-static-cuda-12-1.0.*" "libcudss1-devel-cuda-12-1.0.*" "libcudss1-cuda-12-1.0.*"
        register_cudss_alternatives 12
    fi
elif command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y "libcudss-devel=1.0.*"
else
    echo "Neither dnf nor apt-get found. Cannot install cudss dependencies."
    exit 1
fi
