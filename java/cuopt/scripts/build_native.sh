#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${MODULE_DIR}/../.." && pwd)"
CUOPT_PREFIX="${CUOPT_PREFIX:-${CONDA_PREFIX:-${REPO_ROOT}/.cuopt_env}}"
BUILD_DIR="${CUOPT_JAVA_NATIVE_BUILD_DIR:-${MODULE_DIR}/build/native}"

if [[ -z "${JAVA_HOME:-}" ]]; then
  JAVAC_PATH="$(command -v javac || true)"
  if [[ -n "${JAVAC_PATH}" ]]; then
    JAVA_HOME="$(dirname "$(dirname "$(readlink -f "${JAVAC_PATH}")")")"
    export JAVA_HOME
  fi
fi

if [[ ! -x "${JAVA_HOME:-}/bin/javac" ]]; then
  echo "JAVA_HOME must point to a JDK containing bin/javac (Java 11 is required)." >&2
  exit 1
fi

CMAKE="${CMAKE:-}"
if [[ -z "${CMAKE}" && -x "${CUOPT_PREFIX}/bin/cmake" ]]; then
  CMAKE="${CUOPT_PREFIX}/bin/cmake"
fi
if [[ -z "${CMAKE}" ]]; then
  CMAKE="$(command -v cmake || true)"
fi
if [[ -z "${CMAKE}" ]]; then
  echo "cmake was not found; use the cuOpt conda environment or set CMAKE." >&2
  exit 1
fi

if [[ ! -f "${CUOPT_PREFIX}/lib/libcuopt.so" ]]; then
  echo "cuOpt shared library was not found at ${CUOPT_PREFIX}/lib/libcuopt.so." >&2
  exit 1
fi

CXX_COMPILER="${CXX:-}"
if [[ -z "${CXX_COMPILER}" && -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  CACHED_CXX_COMPILER="$(sed -n 's/^CMAKE_CXX_COMPILER:.*=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)"
  if [[ -x "${CACHED_CXX_COMPILER}" ]]; then
    CXX_COMPILER="${CACHED_CXX_COMPILER}"
  fi
fi
if [[ -z "${CXX_COMPILER}" && -x "${CUOPT_PREFIX}/bin/c++" ]]; then
  CXX_COMPILER="${CUOPT_PREFIX}/bin/c++"
fi

env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
  "${CMAKE}" -S "${MODULE_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
  -DCUOPT_PREFIX="${CUOPT_PREFIX}" \
  -DCUOPT_RUNTIME_LIBRARY_DIR="${CUOPT_PREFIX}/lib" \
  ${CXX_COMPILER:+-DCMAKE_CXX_COMPILER="${CXX_COMPILER}"} \
  -DJAVA_HOME="${JAVA_HOME}"

env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
  "${CMAKE}" --build "${BUILD_DIR}" --target cuopt_jni \
  --parallel "${PARALLEL_LEVEL:-2}"

echo "Built ${BUILD_DIR}/libcuopt_jni.so"
