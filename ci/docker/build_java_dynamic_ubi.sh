#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# UBI10 counterpart to build_java_dynamic.sh: same idea (build libcuopt_jni.so against the
# already pip-installed libcuopt.so, compile plain classes with javac, no Maven), but RHEL's
# platlib/purelib split puts libcuopt.so under lib64/python3.14/site-packages rather than
# Debian's single dist-packages tree (see Dockerfile.ubi's cuopt-final stage for the same
# lib64-vs-lib distinction on the Python side). Run from the repo root staged under REPO_ROOT
# (see build_images.yaml); writes cuopt.jar and libcuopt_jni.so to OUT_DIR.

set -euo pipefail

REPO_ROOT="${1:?missing repo root}"
OUT_DIR="${2:?missing output directory}"

CUOPT_SITE_PACKAGES="/usr/local/lib64/python3.14/site-packages/libcuopt"
if [[ ! -f "${CUOPT_SITE_PACKAGES}/lib64/libcuopt.so" ]]; then
  echo "libcuopt.so not found under ${CUOPT_SITE_PACKAGES}/lib64; is libcuopt pip-installed yet?" >&2
  exit 1
fi

# See build_java_dynamic.sh for why only CUOPT_LIBRARY/CUOPT_RUNTIME_LIBRARY_DIR need pointing
# at libcuopt.so directly: it resolves TBB/NCCL/cuDSS/rmm/rapids_logger/cublas/... itself via its
# own $ORIGIN-relative RPATH, so cuopt_jni.so only has to find libcuopt.so.
export CUOPT_PREFIX="${CUOPT_SITE_PACKAGES}"
export CUOPT_LIBRARY="${CUOPT_SITE_PACKAGES}/lib64/libcuopt.so"
export CUOPT_RUNTIME_LIBRARY_DIR="${CUOPT_SITE_PACKAGES}/lib64"
# rmm and rapids_logger are separate pip packages with their own include directories, also under
# lib64/python3.14/site-packages on UBI (platlib), unlike a conda install where
# CUOPT_PREFIX/include/rapids covers all three (java/cuopt/CMakeLists.txt's CUOPT_PREFIX/include
# /rapids handling does not apply here).
PIP_SITE_PACKAGES="$(dirname "${CUOPT_SITE_PACKAGES}")"
export CUOPT_EXTRA_INCLUDE_DIRS="${REPO_ROOT}/cpp/include;${REPO_ROOT}/cpp/src;${PIP_SITE_PACKAGES}/librmm/include;${PIP_SITE_PACKAGES}/rapids_logger/include"
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
