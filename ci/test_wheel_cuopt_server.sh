#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "${CUOPT_MPS_PARSER_WHEELHOUSE}"/cuopt_mps_parser*.whl \
    "$(echo "${CUOPT_SERVER_WHEELHOUSE}"/cuopt_server*.whl)[test]" \
    "${CUOPT_WHEELHOUSE}"/cuopt*.whl \
    "${CUOPT_SH_CLIENT_WHEELHOUSE}"/cuopt_sh_client*.whl \
    "${LIBCUOPT_WHEELHOUSE}"/libcuopt*.whl

./datasets/linear_programming/download_pdlp_test_dataset.sh
./datasets/mip/download_miplib_test_dataset.sh

RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR

timeout 30m ./ci/run_cuopt_server_pytests.sh --verbose --capture=no

# Run documentation tests
./ci/test_doc_examples.sh
