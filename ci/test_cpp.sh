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

rapids-logger "Download datasets"
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

rapids-logger "Run gtests"
for gt in "$CONDA_PREFIX"/bin/gtests/libcuopt/*_TEST ; do
    test_name=$(basename "${gt}")
    echo "Running gtest $test_name"
    timeout 20m "${gt}" --gtest_output=xml:"{RAPIDS_TESTS_DIR}"
done

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
