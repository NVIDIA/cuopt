#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds a self-contained Java classifier JAR and checks that it is actually self-contained.
#
# Unlike ci/build_java.sh, which installs a prebuilt libcuopt and links it as a shared library,
# this compiles libcuopt from source as a static archive and embeds it, so the JAR is the only
# thing a consumer installs. See #1817.

set -euo pipefail

if [[ -e /opt/conda/etc/profile.d/conda.sh ]]; then
  . /opt/conda/etc/profile.d/conda.sh
fi

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generating Java static build dependencies"
ENV_YAML_DIR=$(mktemp -d)
rapids-dependency-file-generator \
  --output conda \
  --file-key java_static \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java_static

# Temporarily allow unbound variables for conda activation.
set +u
conda activate java_static
set -u

rapids-print-env

export CUOPT_PREFIX="${CONDA_PREFIX}"
STATIC_BUILD_DIR="${PWD}/cpp/build-static"
JNI_BUILD_DIR="${PWD}/java/cuopt/build/native-static"
JAR_OUTPUT_DIR="${PWD}/java/cuopt/classifier-jars"
MAVEN_REPO_DIR="${PWD}/java/cuopt/maven-repo"

rapids-logger "Building the scoped static libcuopt"
BUILD_DIR="${STATIC_BUILD_DIR}" bash java/cuopt/ci/build_static_libcuopt.sh

rapids-logger "Linking libcuopt into cuopt_jni"
cmake -S java/cuopt -B "${JNI_BUILD_DIR}" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUOPT_PREFIX="${CUOPT_PREFIX}" \
  -DCUOPT_STATIC_BUILD_DIR="${STATIC_BUILD_DIR}" \
  -DCUOPT_EXTRA_INCLUDE_DIRS="${PWD}/cpp/include;${STATIC_BUILD_DIR}/include"
cmake --build "${JNI_BUILD_DIR}" --parallel "${PARALLEL_LEVEL:-$(nproc)}"

rapids-logger "Packaging the classifier JAR"
bash java/cuopt/ci/build_cuopt_java_jar.sh \
  --native-lib "${JNI_BUILD_DIR}/libcuopt_jni.so" \
  --cuda-version "${RAPIDS_CUDA_VERSION}" \
  --output-dir "${JAR_OUTPUT_DIR}"

# The JAR looking fine on this machine proves nothing: the build environment supplies every
# dependency by construction. This resolves them the way a consumer's machine would.
rapids-logger "Verifying the JAR is self-contained"
CLASSIFIER_JAR=$(find "${JAR_OUTPUT_DIR}" -name 'cuopt-*.jar' -print -quit)
bash java/cuopt/ci/verify_jar_dependencies.sh --jar "${CLASSIFIER_JAR}"

# A single artifact in Maven repository layout is what a publishing workflow consumes, so the
# shape is fixed here rather than left to whatever downloads these JARs.
rapids-logger "Assembling the Maven repository layout"
bash java/cuopt/ci/assemble_maven_repo.sh \
  --jars-dir "${JAR_OUTPUT_DIR}" \
  --extra-jars-dir "${PWD}/java/cuopt/target" \
  --output-dir "${MAVEN_REPO_DIR}"

rapids-logger "Result"
du -h "${CLASSIFIER_JAR}" | sed 's/^/  /'
