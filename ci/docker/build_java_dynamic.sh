#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds libcuopt_jni.so against the pip-installed libcuopt.so and compiles the Java classes with
# javac (java/cuopt/pom.xml has no non-test dependency, so Maven isn't needed). Writes cuopt.jar
# and libcuopt_jni.so to OUT_DIR.

set -euo pipefail

REPO_ROOT="${1:?missing repo root}"
CUOPT_SITE_PACKAGES="${2:?missing path to the pip-installed libcuopt package, e.g. /usr/local/lib/python3.14/dist-packages/libcuopt}"
OUT_DIR="${3:?missing output directory}"

if [[ ! -f "${CUOPT_SITE_PACKAGES}/lib64/libcuopt.so" ]]; then
  echo "libcuopt.so not found under ${CUOPT_SITE_PACKAGES}/lib64; is libcuopt pip-installed yet?" >&2
  exit 1
fi

# libcuopt.so is a thin metapackage that DT_NEEDs libcuopt_client.so, libcuopt_mathopt.so, and
# libcuopt_routing.so from their own sibling pip packages, so all four lib64 dirs are needed.
PIP_SITE_PACKAGES="$(dirname "${CUOPT_SITE_PACKAGES}")"
export CUOPT_PREFIX="${CUOPT_SITE_PACKAGES}"
export CUOPT_LIBRARY="${CUOPT_SITE_PACKAGES}/lib64/libcuopt.so"
export CUOPT_RUNTIME_LIBRARY_DIR="${CUOPT_SITE_PACKAGES}/lib64;${PIP_SITE_PACKAGES}/libcuopt_client/lib64;${PIP_SITE_PACKAGES}/libcuopt_mathopt/lib64;${PIP_SITE_PACKAGES}/libcuopt_routing/lib64"
# rmm, rapids_logger, and raft are separate pip packages with their own include dirs (unlike
# conda, where CUOPT_PREFIX/include/rapids covers all three). raft needs <cuda/std/mdspan>,
# which ships in nvidia-cuda-cccl: nvidia/cuXY/include/cccl on CUDA 13, nvidia/cuda_cccl/include
# on CUDA 12. nvidia-cuda-cccl is purelib (installs to Python's "lib" site-packages); libcuopt
# is platlib ("lib64" on RHEL/UBI, same as "lib" on Debian) -- check both, see Dockerfile.ubi.
PURELIB_SITE_PACKAGES="${PIP_SITE_PACKAGES/\/lib64\//\/lib\/}"
CCCL_INCLUDE_DIR="$(compgen -G "${PIP_SITE_PACKAGES}/nvidia/cu*/include/cccl" || true)"
if [[ -z "${CCCL_INCLUDE_DIR}" ]]; then
  CCCL_INCLUDE_DIR="$(compgen -G "${PURELIB_SITE_PACKAGES}/nvidia/cu*/include/cccl" || true)"
fi
CCCL_INCLUDE_DIR="$(head -1 <<<"${CCCL_INCLUDE_DIR}")"
if [[ -z "${CCCL_INCLUDE_DIR}" ]]; then
  for candidate in "${PIP_SITE_PACKAGES}/nvidia/cuda_cccl/include" "${PURELIB_SITE_PACKAGES}/nvidia/cuda_cccl/include"; do
    if [[ -d "${candidate}" ]]; then
      CCCL_INCLUDE_DIR="${candidate}"
      break
    fi
  done
fi
if [[ ! -d "${CCCL_INCLUDE_DIR}" ]]; then
  echo "nvidia-cuda-cccl include dir not found under ${PIP_SITE_PACKAGES}/nvidia or ${PURELIB_SITE_PACKAGES}/nvidia" >&2
  exit 1
fi
export CUOPT_EXTRA_INCLUDE_DIRS="${REPO_ROOT}/cpp/include;${REPO_ROOT}/cpp/src;${PIP_SITE_PACKAGES}/librmm/include;${PIP_SITE_PACKAGES}/rapids_logger/include;${PIP_SITE_PACKAGES}/libraft/include;${CCCL_INCLUDE_DIR}"
export CUOPT_JAVA_NATIVE_BUILD_DIR="${REPO_ROOT}/java/cuopt/build/native"

cd "${REPO_ROOT}"
bash java/cuopt/scripts/build_native.sh

GEN_SRC_DIR="${REPO_ROOT}/java/cuopt/target/generated-sources/cuopt"
bash java/cuopt/scripts/generate_constants.sh \
  "${REPO_ROOT}/cpp/include/cuopt/mathematical_optimization/constants.h" \
  "${GEN_SRC_DIR}"

CLASSES_DIR="$(mktemp -d)"
mapfile -t JAVA_SOURCES < <(find "${REPO_ROOT}/java/cuopt/src/main/java" "${GEN_SRC_DIR}" -name '*.java')
javac -d "${CLASSES_DIR}" --release 17 "${JAVA_SOURCES[@]}"

mkdir -p "${OUT_DIR}"
jar cf "${OUT_DIR}/cuopt.jar" -C "${CLASSES_DIR}" .
cp "${CUOPT_JAVA_NATIVE_BUILD_DIR}/libcuopt_jni.so" "${OUT_DIR}/"
rm -rf "${CLASSES_DIR}"

echo "Wrote ${OUT_DIR}/cuopt.jar and ${OUT_DIR}/libcuopt_jni.so"
