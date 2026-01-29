#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX="true"
export RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX
source rapids-init-pip

# Install rockylinux repo
if command -v dnf &> /dev/null; then
    bash ci/utils/update_rockylinux_repo.sh
fi

# Install cudss
bash ci/utils/install_cudss.sh

package_dir="python/cuopt"
export SKBUILD_CMAKE_ARGS="-DCUOPT_BUILD_WHEELS=ON;-DDISABLE_DEPRECATION_WARNINGS=ON";

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the libcuopt and cuopt-mps-parser wheel built in the previous step and make it
# available for pip to find.
#
# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except those used in 'pip wheel', handled separately in build scripts)
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUOPT_MPS_PARSER_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_mps_parser" rapids-download-wheels-from-github python)
LIBCUOPT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

echo "libcuopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUOPT_WHEELHOUSE}/libcuopt_*.whl)" >> "${PIP_CONSTRAINT}"
echo "cuopt-mps-parser @ file://$(echo ${CUOPT_MPS_PARSER_WHEELHOUSE}/cuopt_mps_parser*.whl)" >> "${PIP_CONSTRAINT}"

EXCLUDE_ARGS=(
  --exclude "libraft.so"
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcuda.so.1"
  --exclude "libcudss.so.*"
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
