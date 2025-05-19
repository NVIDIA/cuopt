#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -euo pipefail
package_dir="python/cuopt"
export SKBUILD_CMAKE_ARGS="-DCUOPT_BUILD_WHEELS=ON;-DDISABLE_DEPRECATION_WARNINGS=ON";

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the libcuopt and cuopt-mps-parser wheel built in the previous step and make it
# available for pip to find.
# Download the packages built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUOPT_MPS_PARSER_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_mps_parser" rapids-download-wheels-from-github python)
LIBCUOPT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

echo "libcuopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUOPT_WHEELHOUSE}/libcuopt_*.whl)" >> /tmp/constraints.txt
echo "cuopt-mps-parser @ file://$(echo ${CUOPT_MPS_PARSER_WHEELHOUSE}/cuopt_mps_parser*.whl)" >> /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"


EXCLUDE_ARGS=(
  --exclude "libraft.so"
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libcuopt.so"
  --exclude "libmps_parser.so"
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
)

ci/build_wheel.sh cuopt ${package_dir}

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair "${EXCLUDE_ARGS[@]}" -w ${RAPIDS_WHEEL_BLD_OUTPUT_DIR} ${package_dir}/dist/*

ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PY_WHEEL_NAME="cuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
