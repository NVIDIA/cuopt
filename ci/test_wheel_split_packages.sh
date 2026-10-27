#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installs each libcuopt-* wheel on its own, in a fresh venv, and builds and runs a C++ binary
# against it, to catch a wheel that only works because another one happened to be installed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPONENTS=(client mathopt routing)

source rapids-init-pip
bash "${REPO_ROOT}/ci/utils/install_openssl3_runtime.sh"

# find_package(cuopt) needs CMake's FindCUDAToolkit, which wants nvcc and the CUDA dev files
# that the test image does not ship.
CUDA_PKG_SUFFIX="$(echo "${RAPIDS_CUDA_VERSION}" | cut -d. -f1,2 | tr . -)"
if command -v apt-get &> /dev/null; then
  DEV=dev
  INSTALL=(apt-get install -y)
  apt-get update
else
  DEV=devel
  INSTALL=(dnf install -y)
fi
"${INSTALL[@]}" \
  "cuda-nvcc-${CUDA_PKG_SUFFIX}" \
  "cuda-cudart-${DEV}-${CUDA_PKG_SUFFIX}" \
  "libcublas-${DEV}-${CUDA_PKG_SUFFIX}" \
  "libcusparse-${DEV}-${CUDA_PKG_SUFFIX}" \
  "libcusolver-${DEV}-${CUDA_PKG_SUFFIX}" \
  "libcurand-${DEV}-${CUDA_PKG_SUFFIX}"
export PATH="/usr/local/cuda/bin:${PATH}"

SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CLIENT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_client cuopt)")
MATHOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_mathopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
ROUTING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_routing cuopt --cuda "$RAPIDS_CUDA_VERSION")")

cat > "${PIP_CONSTRAINT}" <<CONSTRAINTS
libcuopt-client @ file://$(echo "${CLIENT_WHEELHOUSE}"/libcuopt_client-*.whl)
libcuopt-mathopt-${SUFFIX} @ file://$(echo "${MATHOPT_WHEELHOUSE}"/libcuopt_mathopt_*.whl)
libcuopt-routing-${SUFFIX} @ file://$(echo "${ROUTING_WHEELHOUSE}"/libcuopt_routing_*.whl)
CONSTRAINTS

EXITCODE=0
trap "EXITCODE=1" ERR

for component in "${COMPONENTS[@]}"; do
  echo "=== wheel: libcuopt-${component} alone ==="

  if [[ "${component}" == "client" ]]; then
    package=libcuopt-client
    expected=(libcuopt-client)
  else
    package="libcuopt-${component}-${SUFFIX}"
    expected=(libcuopt-client "${package}")
  fi

  venv="$(mktemp -d)/venv"
  set +e
  python -m venv "${venv}"
  . "${venv}/bin/activate"
  pip install --upgrade pip cmake ninja
  rapids-pip-retry install --prefer-binary --constraint "${PIP_CONSTRAINT}" "${package}"

  python "${REPO_ROOT}/ci/split_package_smoke/env_check.py" wheel-packages "${expected[@]}"
  prefixes=$(python "${REPO_ROOT}/ci/split_package_smoke/env_check.py" cmake-prefixes)
  bash "${REPO_ROOT}/ci/split_package_smoke/build_and_run.sh" "${component}" "${prefixes}"
  deactivate
  set -e
done

rapids-logger "Test script exiting with value: ${EXITCODE}"
exit "${EXITCODE}"
