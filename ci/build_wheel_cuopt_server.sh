#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_dir="python/cuopt_server"

ci/build_wheel.sh cuopt_server ${package_dir}
cp "${package_dir}/dist"/* "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/"
ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

# `cuopt-server` is a pure + CUDA version package
RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python cuopt-server cuopt --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
