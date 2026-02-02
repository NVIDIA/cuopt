#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX="true"
export RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX
source rapids-init-pip

package_dir="python/cuopt/cuopt/linear_programming"

export SKBUILD_CMAKE_ARGS="-DCUOPT_BUILD_WHEELS=ON;-DDISABLE_DEPRECATION_WARNINGS=ON"

if [ "$RAPIDS_BUILD_TYPE" = "pull-request" ]; then
    echo "Building in assert mode"
    export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};-DDEFINE_ASSERT=True"
else
    echo "Building in release mode"
fi

ci/build_wheel.sh cuopt_mps_parser ${package_dir}


EXCLUDE_ARGS=(
  --exclude "libzlib.so"
  --exclude "libbz2.so"
)

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair "${EXCLUDE_ARGS[@]}" -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" ${package_dir}/dist/*

ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
