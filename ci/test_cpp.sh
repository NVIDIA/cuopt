#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test --channel "${CPP_CHANNEL}"

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Download datasets"

# Debug: Check if S3 configuration is available
if [ -n "${CUOPT_DATASET_S3_URI:-}" ]; then
    echo "✓ CUOPT_DATASET_S3_URI is set: $CUOPT_DATASET_S3_URI"
    export CUOPT_DATASET_S3_URI
else
    echo "✗ CUOPT_DATASET_S3_URI not set"
fi

if [ -n "${CUOPT_AWS_ACCESS_KEY_ID:-}" ]; then
    echo "✓ CUOPT_AWS_ACCESS_KEY_ID is available (via secrets: inherit)"
    export CUOPT_AWS_ACCESS_KEY_ID
    export CUOPT_AWS_SECRET_ACCESS_KEY
else
    echo "✗ CUOPT_AWS_ACCESS_KEY_ID not found in environment"
    echo "  S3 download will fall back to HTTP if configured"
fi
./datasets/linear_programming/download_pdlp_test_dataset.sh
./datasets/mip/download_miplib_test_dataset.sh

RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR
pushd "${RAPIDS_DATASET_ROOT_DIR}"
./get_test_data.sh
popd

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run gtests from libcuopt-tests package
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

rapids-logger "Run gtests"
timeout 40m ./ci/run_ctests.sh

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
