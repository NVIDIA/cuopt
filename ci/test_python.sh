#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

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

# Due to race condition in certain cases UCX might not be able to cleanup properly, so we set the number of threads to 1
export OMP_NUM_THREADS=1

rapids-logger "Test cuopt_cli"
timeout 10m bash ./python/libcuopt/libcuopt/tests/test_cli.sh

rapids-logger "pytest cuopt"
timeout 30m ./ci/run_cuopt_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuopt.xml" \
  --cov-config=.coveragerc \
  --cov=cuopt \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuopt-coverage.xml" \
  --cov-report=term \
  --ignore=raft

rapids-logger "pytest cuopt-server"
timeout 20m ./ci/run_cuopt_server_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuopt-server.xml" \
  --cov-config=.coveragerc \
  --cov=cuopt_server \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuopt-server-coverage.xml" \
  --cov-report=term

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
