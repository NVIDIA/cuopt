#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

source rapids-init-pip

# Download the packages built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUOPT_MPS_PARSER_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_mps_parser" rapids-download-wheels-from-github python)
CUOPT_SERVER_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_server_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)
CUOPT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
CUOPT_SH_CLIENT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_sh_client" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)
LIBCUOPT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    --extra-index-url=https://pypi.nvidia.com \
    "${CUOPT_MPS_PARSER_WHEELHOUSE}"/cuopt_mps_parser*.whl \
    "$(echo "${CUOPT_SERVER_WHEELHOUSE}"/cuopt_server*.whl)[test]" \
    "${CUOPT_WHEELHOUSE}"/cuopt*.whl \
    "${CUOPT_SH_CLIENT_WHEELHOUSE}"/cuopt_sh_client*.whl \
    "${LIBCUOPT_WHEELHOUSE}"/libcuopt*.whl

# Install sanitizer libraries for runtime
if command -v apt-get &> /dev/null; then
    apt-get -y update
    apt-get -y install libasan8 libubsan1
elif command -v dnf &> /dev/null; then
    dnf -y update
    # Install sanitizer libraries from gcc-toolset if available
    if dnf list installed gcc-toolset-14 &> /dev/null; then
        dnf -y install gcc-toolset-14-libasan-devel gcc-toolset-14-libubsan-devel
    elif dnf list installed gcc-toolset-13 &> /dev/null; then
        dnf -y install gcc-toolset-13-libasan-devel gcc-toolset-13-libubsan-devel
    else
        dnf -y install libasan libubsan
    fi
fi

./datasets/linear_programming/download_pdlp_test_dataset.sh
./datasets/mip/download_miplib_test_dataset.sh


RAPIDS_DATASET_ROOT_DIR=./datasets timeout 30m python -m pytest --verbose --capture=no ./python/cuopt_server/cuopt_server/tests/

# Run documentation tests
./ci/test_doc_examples.sh
