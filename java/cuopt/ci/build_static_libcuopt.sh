#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds libcuopt_static.a scoped to what the Java bindings actually expose, and reports its
# size. See #1817.
#
# The Java API covers LP, MIP and QP only, so routing, the gRPC server and NCCL's distributed
# PDLP path are all excluded. That matters because the shared libcuopt is 554 MB against 29
# DT_NEEDED entries, and Maven Central caps an upload bundle at 1 GB — a self-contained JAR is
# only viable if the embedded library is scoped first.
#
# This script does not produce a JAR. It exists to measure whether one is feasible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/cpp/build-static}"
PARALLEL_LEVEL="${PARALLEL_LEVEL:-$(nproc)}"
CUDA_ARCHS="${CUOPT_CMAKE_CUDA_ARCHITECTURES:-RAPIDS}"

# Routing and gRPC are excluded here rather than in a Java-specific fork of the build, because
# cpp/CMakeLists.txt already offers the switches.
cmake_args=(
  -S "${REPO_ROOT}/cpp"
  -B "${BUILD_DIR}"
  -GNinja
  -DCMAKE_BUILD_TYPE=Release
  -DCUOPT_BUILD_STATIC_LIB=ON
  -DBUILD_TESTS=OFF
  -DSKIP_ROUTING_BUILD=ON
  -DSKIP_GRPC_BUILD=ON
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"
)

echo "Configuring scoped static build in ${BUILD_DIR}"
cmake "${cmake_args[@]}"

echo "Building cuopt_static with ${PARALLEL_LEVEL} jobs"
cmake --build "${BUILD_DIR}" --target cuopt_static --parallel "${PARALLEL_LEVEL}"

archive="$(find "${BUILD_DIR}" -name 'libcuopt_static.a' -print -quit)"
if [[ -z "${archive}" ]]; then
  echo "cuopt_static built but libcuopt_static.a was not found under ${BUILD_DIR}" >&2
  exit 1
fi

# The archive is an upper bound, not the shipped size: linking it into a shared object keeps
# only the objects that are actually referenced.
size_mb=$(( $(stat -c%s "${archive}") / 1048576 ))
echo
echo "  archive : ${archive}"
echo "  size    : ${size_mb} MB (unlinked upper bound)"
echo
echo "Link this into libcuopt_jni.so to get the figure that decides whether a"
echo "self-contained classifier JAR fits inside the 1 GB Maven Central bundle limit."
