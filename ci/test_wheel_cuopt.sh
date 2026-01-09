#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# sets up a constraints file for 'pip' and puts its location in an exported variable PIP_EXPORT,
# so those constraints will affect all future 'pip install' calls
source rapids-init-pip

# Download the packages built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
#UOPT_MPS_PARSER_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_mps_parser" rapids-download-wheels-from-github python)
#UOPT_SH_CLIENT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_sh_client" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)
#UOPT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
#IBCUOPT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# update pip constraints.txt to ensure all future 'pip install' (including those in ci/thirdparty-testing)
# use these wheels for cuopt packages
#at > "${PIP_CONSTRAINT}" <<EOF
#uopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${CUOPT_WHEELHOUSE}/cuopt_${RAPIDS_PY_CUDA_SUFFIX}-*.whl)
#uopt-mps-parser @ file://$(echo ${CUOPT_MPS_PARSER_WHEELHOUSE}/cuopt_mps_parser-*.whl)
#uopt-sh-client @ file://$(echo ${CUOPT_SH_CLIENT_WHEELHOUSE}/cuopt_sh_client-*.whl)
#ibcuopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUOPT_WHEELHOUSE}/libcuopt_${RAPIDS_PY_CUDA_SUFFIX}-*.whl)
#OF

# echo to expand wildcard before adding `[extra]` requires for pip
#apids-pip-retry install \
#   --extra-index-url=https://pypi.nvidia.com \
#   --constraint "${PIP_CONSTRAINT}" \
#   "${CUOPT_MPS_PARSER_WHEELHOUSE}"/cuopt_mps_parser*.whl \
#   "$(echo "${CUOPT_WHEELHOUSE}"/cuopt*.whl)[test]" \
#   "${CUOPT_SH_CLIENT_WHEELHOUSE}"/cuopt_sh_client*.whl \
#   "${LIBCUOPT_WHEELHOUSE}"/libcuopt*.whl

#ython -c "import cuopt"
#
#f command -v apt-get &> /dev/null; then
#   apt-get -y update
#   apt-get -y install file unzip
#lif command -v dnf &> /dev/null; then
#   dnf -y update
#   dnf -y install file unzip
#fi

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
cd ./datasets
./get_test_data.sh --solomon
./get_test_data.sh --tsp
cd -

RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR

# Run CLI tests
#timeout 10m bash ./python/libcuopt/libcuopt/tests/test_cli.sh

# Run Python tests

# Due to race condition in certain cases UCX might not be able to cleanup properly, so we set the number of threads to 1
#export OMP_NUM_THREADS=1

#timeout 30m ./ci/run_cuopt_pytests.sh --verbose --capture=no

# run thirdparty integration tests for only nightly builds
# [[ "${RAPIDS_BUILD_TYPE}" == "nightly" ]]; then
    #./ci/thirdparty-testing/run_jump_tests.sh
    #./ci/thirdparty-testing/run_cvxpy_tests.sh
#i
