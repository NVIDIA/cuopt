#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Development script to run the cuopt remote server with the correct library path
# This is only needed during development when the build directory's libcuopt.so
# needs to take precedence over the conda-installed version.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"

if [ ! -f "${BUILD_DIR}/cuopt_remote_server" ]; then
    echo "Error: cuopt_remote_server not found. Please build with -DBUILD_REMOTE_SERVER=ON"
    exit 1
fi

export LD_LIBRARY_PATH="${BUILD_DIR}:${LD_LIBRARY_PATH}"

echo "Starting cuopt remote server..."
echo "Build directory: ${BUILD_DIR}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "---"

exec "${BUILD_DIR}/cuopt_remote_server" "$@"
