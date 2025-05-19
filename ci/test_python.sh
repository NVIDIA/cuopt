#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

CUOPT_VERSION="$(rapids-version)"

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  "libcuopt=${CUOPT_VERSION}" \
  "cuopt=${CUOPT_VERSION}" \
  "cuopt-server=${CUOPT_VERSION}"

rapids-logger "Download datasets"
RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR
./datasets/linear_programming/download_pdlp_test_dataset.sh
./datasets/mip/download_miplib_test_dataset.sh
pushd "${RAPIDS_DATASET_ROOT_DIR}"
./get_test_data.sh
popd

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuopt"
pushd python/cuopt/cuopt
timeout 30m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuopt.xml" \
  --cov-config=.coveragerc \
  --cov=cuopt \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuopt-coverage.xml" \
  --cov-report=term \
  --ignore=raft \
  tests
popd

rapids-logger "pytest cuopt-server"
pushd python/cuopt_server/cuopt_server
timeout 20m pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuopt-server.xml" \
  --cov-config=.coveragerc \
  --cov=cuopt_server \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuopt-server-coverage.xml" \
  --cov-report=term \
  tests
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
