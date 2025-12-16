#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [[ "$(date +%A)" != "Friday" ]]; then
  echo "Not Friday, exiting early."
  exit 0
fi

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

RAPIDS_VERSION="$(rapids-version)"

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  "libcuopt=${RAPIDS_VERSION}" \
  "libcuopt-tests=${RAPIDS_VERSION}"

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Download datasets"
RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR
pushd "${RAPIDS_DATASET_ROOT_DIR}"
./get_test_data.sh
popd

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run gtests with compute sanitizer"
checkers=( "memcheck" "synccheck" "racecheck" )
tests=( "ROUTING_TEST" "ROUTING_GES_TEST" )
for j in "${tests[@]}" ; do
    for i in "${checkers[@]}" ; do
        gt="$CONDA_PREFIX"/bin/gtests/libcuopt/"$j"
        test_name=$(basename "${gt}")
        echo "Running gtest with compute sanitizer --tool $i $test_name"
        COMPUTE_SANITIZER_CMD="compute-sanitizer --tool ${i}"
        ${COMPUTE_SANITIZER_CMD} "${gt}" --gtest_output=xml:"${RAPIDS_TESTS_DIR}${test_name}${i}.xml"
    done
done

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
