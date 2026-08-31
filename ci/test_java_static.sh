#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs the Java test suite against an already-packaged classifier JAR, on a GPU, with no
# libcuopt installed. See #1817.
#
# ci/build_java_static.sh checks the JAR's dependencies statically; this is the other half,
# that the libraries it carries actually load and produce correct answers.
#
# Activates -Ppackaged-jar-tests so main compilation is skipped and the JAR supplies the classes
# and the native libraries. PackagedJarOriginCheck then asserts that is genuinely where they came
# from, so a stray target/classes cannot make this pass while testing the wrong thing.
#
# CUOPT_JAVA_JAR may be set to a classifier JAR to skip the download and test it directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=java/cuopt/ci/java_classifier.sh
. "${REPO_ROOT}/java/cuopt/ci/java_classifier.sh"
# shellcheck source=java/cuopt/scripts/maven.sh
. "${REPO_ROOT}/java/cuopt/scripts/maven.sh"
cuopt_maven_args

if [[ -e /opt/conda/etc/profile.d/conda.sh ]]; then
  . /opt/conda/etc/profile.d/conda.sh
fi

if [[ -z "${CUOPT_JAVA_JAR:-}" ]]; then
  case "$(arch)" in
    x86_64) JOB_ARCH=amd64 ;;
    aarch64) JOB_ARCH=arm64 ;;
    *) echo "unsupported architecture $(arch)" >&2; exit 1 ;;
  esac
  ARTIFACT="cuopt_java_${JOB_ARCH}_cu${RAPIDS_CUDA_VERSION%%.*}"
  rapids-logger "Downloading ${ARTIFACT}"
  JAVA_PKG="$(rapids-download-from-github "${ARTIFACT}")"
  CUOPT_JAVA_JAR="$(cuopt_java_resolve_artifact_jar "${JAVA_PKG}")"
fi
rapids-logger "Testing $(basename "${CUOPT_JAVA_JAR}")"

# A JDK, Maven and the CUDA runtime only. Installing libcuopt would defeat the test, since the
# JAR is supposed to carry its own copy.
rapids-logger "Creating a consumer-like environment"
ENV_YAML_DIR=$(mktemp -d)
cat > "${ENV_YAML_DIR}/env.yaml" << EOF
name: java_static_test
channels:
  - conda-forge
dependencies:
  - openjdk=11.*
  - maven
  - cuda-version=${RAPIDS_CUDA_VERSION%.*}
  - libcublas
  - libcusparse
EOF
rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java_static_test

set +u
conda activate java_static_test
set -u

if [[ -e "${CONDA_PREFIX}/lib/libcuopt.so" ]]; then
  echo "ERROR: libcuopt.so is present in the test environment, so passing here would not show" >&2
  echo "       that the JAR is self-contained." >&2
  exit 1
fi

rapids-print-env
nvidia-smi

rapids-logger "Running the suite against the packaged JAR"
if ! cuopt_mvn -B -f "${REPO_ROOT}/java/cuopt/pom.xml" test \
  -Ppackaged-jar-tests \
  "-Dcuopt.jar.path=${CUOPT_JAVA_JAR}"; then
  # Surefire's forked-JVM crash diagnostics (e.g. a raw native write to stdout corrupting its
  # fork-communication channel) land in target/surefire-reports/*.dumpstream and any
  # hs_err_pid*.log a real JVM crash leaves behind. Neither is printed to the console or
  # uploaded as an artifact by this job, so a failure here is otherwise a dead end without
  # reproducing it locally. Print them inline instead.
  rapids-logger "Test failure -- dumping Surefire fork-crash diagnostics"
  find "${REPO_ROOT}/java/cuopt/target/surefire-reports" -type f \
    \( -name '*.dumpstream' -o -name 'hs_err_pid*.log' \) -print0 2>/dev/null |
    while IFS= read -r -d '' f; do
      echo "----- ${f} -----"
      cat "${f}"
    done
  exit 1
fi

rapids-logger "Classifier JAR verified end to end"
