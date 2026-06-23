#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

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

# Download the libcuopt wheel built in the previous step and make it
# available for pip to find.
#
# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except those used in 'pip wheel', handled separately in build scripts)
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBCUOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")

echo "libcuopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUOPT_WHEELHOUSE}/libcuopt_*.whl)" >> "${PIP_CONSTRAINT}"

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
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
)

ci/build_wheel.sh cuopt ${package_dir}

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair "${EXCLUDE_ARGS[@]}" -w ${RAPIDS_WHEEL_BLD_OUTPUT_DIR} ${package_dir}/dist/*

ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python cuopt cuopt --py "$RAPIDS_PY_VERSION" --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
