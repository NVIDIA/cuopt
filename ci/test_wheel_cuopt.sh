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

# sets up a constraints file for 'pip' and puts its location in an exported variable PIP_EXPORT,
# so those constraints will affect all future 'pip install' calls
source rapids-init-pip

# Download the packages built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUOPT_MPS_PARSER_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_mps_parser" rapids-download-wheels-from-github python)
CUOPT_SH_CLIENT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_sh_client" rapids-download-wheels-from-github python)
CUOPT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUOPT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# update pip constraints.txt to ensure all future 'pip install' (including those in ci/thirdparty-testing)
# use these wheels for cuopt packages
cat > "${PIP_CONSTRAINT}" <<EOF
cuopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${CUOPT_WHEELHOUSE}/cuopt_${RAPIDS_PY_CUDA_SUFFIX}-*.whl)
cuopt-mps-parser @ file://$(echo ${CUOPT_MPS_PARSER_WHEELHOUSE}/cuopt_mps_parser-*.whl)
cuopt-sh-client @ file://$(echo ${CUOPT_SH_CLIENT_WHEELHOUSE}/cuopt_sh_client-*.whl)
libcuopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUOPT_WHEELHOUSE}/libcuopt_${RAPIDS_PY_CUDA_SUFFIX}-*.whl)
EOF

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    --extra-index-url=https://pypi.nvidia.com \
    --constraint "${PIP_CONSTRAINT}" \
    "${CUOPT_MPS_PARSER_WHEELHOUSE}"/cuopt_mps_parser*.whl \
    "$(echo "${CUOPT_WHEELHOUSE}"/cuopt*.whl)[test]" \
    "${CUOPT_SH_CLIENT_WHEELHOUSE}"/cuopt_sh_client*.whl \
    "${LIBCUOPT_WHEELHOUSE}"/libcuopt*.whl

python -c "import cuopt"

if command -v apt-get &> /dev/null; then
    apt-get -y update
    apt-get -y install file unzip
elif command -v dnf &> /dev/null; then
    dnf -y update
    dnf -y install file unzip
fi

./datasets/linear_programming/download_pdlp_test_dataset.sh
./datasets/mip/download_miplib_test_dataset.sh
cd ./datasets
./get_test_data.sh --solomon
./get_test_data.sh --tsp
cd -

RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR

# Please enable this once ISSUE https://github.com/NVIDIA/cuopt/issues/94 is fixed
# Run CLI tests
timeout 10m bash ./python/libcuopt/libcuopt/tests/test_cli.sh

# Run Python tests
RAPIDS_DATASET_ROOT_DIR=./datasets timeout 30m python -m pytest --verbose --capture=no ./python/cuopt/cuopt/tests/

# run cvxpy integration tests
./ci/thirdparty-testing/run_cvxpy_tests.sh
