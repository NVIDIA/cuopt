#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Detect which test components to run
source ./ci/detect_test_components.sh

rapids-logger "Download datasets"
RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR

if [[ "${CUOPT_TEST_COMPONENTS}" == "all" || "${CUOPT_TEST_COMPONENTS}" == *"lp"* ]]; then
    ./datasets/linear_programming/download_pdlp_test_dataset.sh
else
    rapids-logger "Skipping LP dataset download (not needed for components: ${CUOPT_TEST_COMPONENTS})"
fi

if [[ "${CUOPT_TEST_COMPONENTS}" == "all" || "${CUOPT_TEST_COMPONENTS}" == *"mip"* ]]; then
    ./datasets/mip/download_miplib_test_dataset.sh
else
    rapids-logger "Skipping MIP dataset download (not needed for components: ${CUOPT_TEST_COMPONENTS})"
fi

if [[ "${CUOPT_TEST_COMPONENTS}" == "all" || "${CUOPT_TEST_COMPONENTS}" == *"routing"* ]]; then
    pushd "${RAPIDS_DATASET_ROOT_DIR}"
    ./get_test_data.sh
    popd
else
    rapids-logger "Skipping routing dataset downloads (not needed for components: ${CUOPT_TEST_COMPONENTS})"
fi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run gtests from libcuopt-tests package
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

rapids-logger "Run gtests (components: ${CUOPT_TEST_COMPONENTS})"
timeout 40m ./ci/run_ctests.sh

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
