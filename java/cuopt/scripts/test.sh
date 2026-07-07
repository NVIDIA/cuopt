#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${MODULE_DIR}/../.." && pwd)"
CUOPT_PREFIX="${CUOPT_PREFIX:-${CONDA_PREFIX:-${REPO_ROOT}/.cuopt_env}}"
NATIVE_BUILD_DIR="${CUOPT_JAVA_NATIVE_BUILD_DIR:-${MODULE_DIR}/build/native}"
export CUOPT_PREFIX CUOPT_JAVA_NATIVE_BUILD_DIR="${NATIVE_BUILD_DIR}"

bash "${MODULE_DIR}/scripts/build_native.sh"

if [[ -z "${JAVA_HOME:-}" ]]; then
  JAVAC_PATH="$(command -v javac || true)"
  if [[ -n "${JAVAC_PATH}" ]]; then
    JAVA_HOME="$(dirname "$(dirname "$(readlink -f "${JAVAC_PATH}")")")"
    export JAVA_HOME
  fi
fi
if [[ ! -x "${JAVA_HOME:-}/bin/java" ]]; then
  echo "JAVA_HOME must point to a JDK containing bin/java (Java 11 is required)." >&2
  exit 1
fi

existing_ld_library_path="${LD_LIBRARY_PATH:-}"
CUDA_RUNTIME_DIR="${CUOPT_PREFIX}/targets/x86_64-linux/lib"
library_path="${CUOPT_PREFIX}/lib:${NATIVE_BUILD_DIR}"
if [[ -d "${CUDA_RUNTIME_DIR}" ]]; then
  library_path="${CUDA_RUNTIME_DIR}:${library_path}"
fi
export LD_LIBRARY_PATH="${library_path}${existing_ld_library_path:+:${existing_ld_library_path}}"

cd "${MODULE_DIR}"
mvn test \
  -Dcuopt.native.dir="${NATIVE_BUILD_DIR}" \
  -Dcuopt.python="${CUOPT_PYTHON:-${CUOPT_PREFIX}/bin/python}" \
  "$@"
