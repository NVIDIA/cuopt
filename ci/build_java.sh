#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build cuopt-java in CI.
#
# Usage:
#   ci/build_java.sh                    # build only, skip tests
#   ci/build_java.sh --run-java-tests   # build + tests
#
# Inputs (set by RAPIDS shared workflow):
#   RAPIDS_CUDA_VERSION
#
# Outputs:
#   java/cuopt-java/target/cuopt-java-*.jar  (uploaded by the calling
#                                             workflow via artifact-name)

set -euo pipefail

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Argument handling — match cuvs build_java.sh contract
RUN_JAVA_TESTS="false"
if [[ "${1:-}" == "--run-java-tests" ]]; then
    RUN_JAVA_TESTS="true"
fi

source rapids-configure-sccache

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

# Pull libcuopt conda package built by upstream conda-cpp-build job.
# libcuopt.so + cuopt_c.h + constants.h end up in $CPP_CHANNEL.
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

rapids-logger "Generate Java conda environment"
rapids-dependency-file-generator \
    --output conda \
    --file-key test_java \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%%.*};arch=$(arch)" \
    --prepend-channel "${CPP_CHANNEL}" \
    | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n java

set +u
conda activate java
set -u

rapids-logger "Build cuopt-java (run_tests=${RUN_JAVA_TESTS})"

# Point the Java build at the conda prefix where libcuopt lives.
export CUOPT_INCLUDE="${CONDA_PREFIX}/include"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}/lib"

# jextract is not on conda-forge. The CI image must provide it.
# TODO: build out a CI image with jextract preinstalled (follow-up PR).
if ! command -v jextract >/dev/null 2>&1; then
    rapids-logger "ERROR: jextract not available in CI image."
    rapids-logger "       Add jextract for JDK 22 to the CI image."
    exit 1
fi

# SKIP_DRIFT_CHECK=true on first build (before panama bindings are committed).
# Remove this once the panama/ files exist in the repo.
if [[ "${RUN_JAVA_TESTS}" == "true" ]]; then
    SKIP_DRIFT_CHECK=true ./java/build.sh
else
    SKIP_DRIFT_CHECK=true SKIP_TESTS=true ./java/build.sh
fi

rapids-logger "Show sccache stats"
sccache --show-adv-stats || true

rapids-logger "Build script exiting with value: $EXITCODE"
exit ${EXITCODE}
