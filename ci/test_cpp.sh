#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

_CUOPT_CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/cuopt_coredumps.sh
source "${_CUOPT_CI_DIR}/cuopt_coredumps.sh"
cuopt_enable_coredumps
trap 'cuopt_collect_coredumps || true' EXIT
unset _CUOPT_CI_DIR
cuopt_coredump_smoke_fail_job_if_enabled

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

set +e

# Run gtests from libcuopt-tests package
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

rapids-logger "Run gtests"
timeout 40m ./ci/run_ctests.sh
EXITCODE=$?
set -e

if [[ "${EXITCODE}" -ne 0 ]]; then
  rapids-logger "run_ctests.sh failed (exit ${EXITCODE}); skipping remaining steps"
fi

rapids-logger "Test script exiting with value: ${EXITCODE}"
exit "${EXITCODE}"
