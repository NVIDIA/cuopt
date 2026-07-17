#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${MODULE_DIR}/../.." && pwd)"
CUOPT_PREFIX="${CUOPT_PREFIX:-${CONDA_PREFIX:-${REPO_ROOT}/.cuopt_env}}"
NATIVE_BUILD_DIR="${CUOPT_JAVA_NATIVE_BUILD_DIR:-${MODULE_DIR}/build/native}"
export CUOPT_PREFIX CUOPT_JAVA_NATIVE_BUILD_DIR="${NATIVE_BUILD_DIR}"

source "${MODULE_DIR}/scripts/java_home.sh"
cuopt_java_setup_home javac

bash "${MODULE_DIR}/scripts/build_native.sh"

cuopt_java_setup_home java

existing_ld_library_path="${LD_LIBRARY_PATH:-}"
CUDA_RUNTIME_DIR="$(find "${CUOPT_PREFIX}/targets" -path "*/lib/libcudart.so" -print -quit 2>/dev/null || true)"
CUDA_RUNTIME_DIR="${CUDA_RUNTIME_DIR%/libcudart.so}"
library_path="${CUOPT_PREFIX}/lib:${NATIVE_BUILD_DIR}"
if [[ -d "${CUDA_RUNTIME_DIR}" ]]; then
  library_path="${CUDA_RUNTIME_DIR}:${library_path}"
fi
export LD_LIBRARY_PATH="${library_path}${existing_ld_library_path:+:${existing_ld_library_path}}"

cd "${MODULE_DIR}"
mvn verify \
  -Dcuopt.native.dir="${NATIVE_BUILD_DIR}" \
  "$@"
