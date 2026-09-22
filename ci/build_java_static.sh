#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds a self-contained Java classifier JAR and checks that it is actually self-contained.
#
# Unlike ci/build_java.sh, which installs a prebuilt libcuopt and links it as a shared library,
# this compiles libcuopt from source as a static archive and embeds it, so the JAR is the only
# thing a consumer installs. See #1817.
#
# Runs in a RAPIDS ci-wheel container rather than ci-conda -- see setup_java_static_env.sh for
# why -- so there is no conda environment here at all. rmm and rapids_logger come from prebuilt
# RAPIDS wheels instead; raft has no such wheel, so it is still CPM-fetched from source by
# cpp/CMakeLists.txt; TBB, cuDSS and NCCL come from dnf/the image itself.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=java/cuopt/ci/setup_java_static_env.sh
. "${REPO_ROOT}/java/cuopt/ci/setup_java_static_env.sh"

rapids-print-env

STATIC_BUILD_DIR="${PWD}/cpp/build-static"
JNI_BUILD_DIR="${PWD}/java/cuopt/build/native-static"
JAR_OUTPUT_DIR="${PWD}/java/cuopt/classifier-jars"

rapids-logger "Building the scoped static libcuopt"
BUILD_DIR="${STATIC_BUILD_DIR}" bash java/cuopt/ci/build_static_libcuopt.sh

# rmm/rapids_logger headers come from the prebuilt wheels' own include directories (found via
# CMAKE_PREFIX_PATH above); raft has no such wheel, so its headers land under the CPM build tree
# instead. Find that one by the source-directory name CPM/FetchContent gives it (<name>-src)
# rather than hardcoding a path that would break the moment a pinned tag/commit changes.
EXTRA_INCLUDE_DIRS="${PWD}/cpp/include;${STATIC_BUILD_DIR}/include"
for dep_include in \
  "${STATIC_BUILD_DIR}/_deps/raft-src/cpp/include" \
  "${STATIC_BUILD_DIR}/_deps/raft-build/include"; do
  if [[ -d "${dep_include}" ]]; then
    EXTRA_INCLUDE_DIRS="${EXTRA_INCLUDE_DIRS};${dep_include}"
  fi
done

rapids-logger "Linking libcuopt into cuopt_jni"
cmake -S java/cuopt -B "${JNI_BUILD_DIR}" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUOPT_STATIC_BUILD_DIR="${STATIC_BUILD_DIR}" \
  -DCUOPT_EXTRA_INCLUDE_DIRS="${EXTRA_INCLUDE_DIRS}" \
  -DCMAKE_PREFIX_PATH="${CUOPT_JAVA_STATIC_CMAKE_PREFIX_PATH:-}"
cmake --build "${JNI_BUILD_DIR}" --parallel "${PARALLEL_LEVEL:-$(nproc)}"

rapids-logger "Packaging the classifier JAR"
# CUOPT_STATIC_BUILD_DIR: where cudss_mtlayer_cuopt.so (built by build_static_libcuopt.sh)
# actually landed, so build_cuopt_java_jar.sh's companion search can find it there -- it is
# dlopen()'d, not linked, so it never appears in libcuopt_jni.so's own DT_NEEDED entries.
export CUOPT_STATIC_BUILD_DIR="${STATIC_BUILD_DIR}"
bash java/cuopt/ci/build_cuopt_java_jar.sh \
  --native-lib "${JNI_BUILD_DIR}/libcuopt_jni.so" \
  --cuda-version "${RAPIDS_CUDA_VERSION}" \
  --output-dir "${JAR_OUTPUT_DIR}"

# The JAR looking fine on this machine proves nothing: the build environment supplies every
# dependency by construction. This resolves them the way a consumer's machine would.
rapids-logger "Verifying the JAR is self-contained"
CLASSIFIER_JAR=$(find "${JAR_OUTPUT_DIR}" -name 'cuopt-*.jar' \
  ! -name '*-sources.jar' ! -name '*-javadoc.jar' -print -quit)
bash java/cuopt/ci/verify_jar_dependencies.sh --jar "${CLASSIFIER_JAR}"

# The gather job combines the classifier directories from every matrix entry into one Maven
# repository layout; this job uploads its own directory as-is.
rapids-logger "Result"
du -h "${CLASSIFIER_JAR}" | sed 's/^/  /'
find "${JAR_OUTPUT_DIR}" -type f | sed "s|^${JAR_OUTPUT_DIR}/|  |" | sort
