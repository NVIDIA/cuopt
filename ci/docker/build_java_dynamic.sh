#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds the Java bindings' native library (libcuopt_jni.so) against the libcuopt.so already
# pip-installed in this image, and compiles the plain (non-classifier) Java classes with javac
# rather than Maven -- java/cuopt/pom.xml has no non-test dependency at all, so there is nothing
# for Maven to resolve, and skipping it avoids adding a Maven Central/mirror network dependency
# to every nightly/branch image build. Run from the repo root staged under REPO_ROOT (see
# build_images.yaml); writes cuopt.jar and libcuopt_jni.so to OUT_DIR.

set -euo pipefail

REPO_ROOT="${1:?missing repo root}"
PYTHON_SHORT_VER="${2:?missing python short version, e.g. 3.14}"
OUT_DIR="${3:?missing output directory}"

CUOPT_SITE_PACKAGES="/usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/libcuopt"
if [[ ! -f "${CUOPT_SITE_PACKAGES}/lib64/libcuopt.so" ]]; then
  echo "libcuopt.so not found under ${CUOPT_SITE_PACKAGES}/lib64; is libcuopt pip-installed yet?" >&2
  exit 1
fi

# build_native.sh (java/cuopt/scripts) already knows how to point the JNI CMake build at an
# arbitrary cuOpt install; only the paths differ from its conda-prefix default. libcuopt.so
# resolves everything else (TBB, NCCL, cuDSS, rmm, rapids_logger, cublas, ...) itself through its
# own $ORIGIN-relative RPATH (see its rpath entries), so cuopt_jni.so only has to find libcuopt.so
# -- CUOPT_RUNTIME_LIBRARY_DIR bakes that into its own RPATH, no LD_LIBRARY_PATH needed.
export CUOPT_PREFIX="${CUOPT_SITE_PACKAGES}"
export CUOPT_LIBRARY="${CUOPT_SITE_PACKAGES}/lib64/libcuopt.so"
export CUOPT_RUNTIME_LIBRARY_DIR="${CUOPT_SITE_PACKAGES}/lib64"
# raft's headers ship bundled inside libcuopt's own include tree (dist-packages/libcuopt/include
# /raft), but rmm and rapids_logger are their own separate pip packages with their own include
# directories -- unlike a conda install, where CUOPT_PREFIX/include/rapids covers all three
# (see the CUOPT_PREFIX/include/rapids handling in java/cuopt/CMakeLists.txt, which does not
# apply here).
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
