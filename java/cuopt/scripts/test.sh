#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${MODULE_DIR}/../.." && pwd)"
CUOPT_PREFIX="${CUOPT_PREFIX:-${CONDA_PREFIX:-${REPO_ROOT}/.cuopt_env}}"
NATIVE_BUILD_DIR="${CUOPT_JAVA_NATIVE_BUILD_DIR:-${MODULE_DIR}/build/native}"
export CUOPT_PREFIX CUOPT_JAVA_NATIVE_BUILD_DIR="${NATIVE_BUILD_DIR}"

source "${MODULE_DIR}/scripts/java_home.sh"
source "${MODULE_DIR}/scripts/maven.sh"
cuopt_java_setup_home javac
cuopt_maven_args

bash "${MODULE_DIR}/scripts/build_native.sh"

cuopt_java_setup_home java

existing_ld_library_path="${LD_LIBRARY_PATH:-}"
CUDA_RUNTIME_DIR="$(find "${CUOPT_PREFIX}/targets" -path "*/lib/libcudart.so" -print -quit 2>/dev/null || true)"
CUDA_RUNTIME_DIR="${CUDA_RUNTIME_DIR%/libcudart.so}"
library_path="${CUOPT_PREFIX}/lib:${NATIVE_BUILD_DIR}"
if [[ -d "${CUDA_RUNTIME_DIR}" ]]; then
  library_path="${CUDA_RUNTIME_DIR}:${library_path}"
fi
if [[ -n "${CUOPT_EXTRA_LIBRARY_DIRS:-}" ]]; then
  library_path="${CUOPT_EXTRA_LIBRARY_DIRS}:${library_path}"
fi
export LD_LIBRARY_PATH="${library_path}${existing_ld_library_path:+:${existing_ld_library_path}}"

# When CPM fetches its own rmm (because the conda prefix has a different version), libcuopt is
# compiled against that copy but records the conda lib dir ahead of the _deps directories in its
# RPATH. DT_RPATH is searched before LD_LIBRARY_PATH, so the loader would pick up the conda
# librmm and fail on rmm's version-tagged inline namespace. Preloading the matching library is
# the only way to win that lookup without relinking libcuopt.
preload="${LD_PRELOAD:-}"
for candidate in ${CUOPT_PRELOAD_LIBS:-}; do
  if [[ -f "${candidate}" ]]; then
    preload="${candidate}${preload:+:${preload}}"
  fi
done
if [[ -n "${preload}" ]]; then
  export LD_PRELOAD="${preload}"
fi

cd "${MODULE_DIR}"
cuopt_mvn verify \
  -Dcuopt.native.dir="${NATIVE_BUILD_DIR}" \
  "$@"
