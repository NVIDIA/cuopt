#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eou pipefail

source rapids-init-pip

# Make sure libssl.so.3 / libcrypto.so.3 are on the linker path before any
# 'import cuopt'. cuopt wheels link OpenSSL 3 and don't bundle it; on Rocky 8
# the runtime needs to come from EPEL (no-op on distros that already ship it).
bash "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/utils/install_openssl3_runtime.sh"

# Download the packages built in the previous step
LIBCUOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
CUOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuopt cuopt --py "$RAPIDS_PY_VERSION" --cuda "$RAPIDS_CUDA_VERSION")")
CUOPT_SERVER_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuopt-server cuopt --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")")
CUOPT_SH_CLIENT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuopt-sh-client cuopt --pure --arch any)")

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
    "$(echo "${CUOPT_SERVER_WHEELHOUSE}"/cuopt_server*.whl)[test]" \
    "${CUOPT_WHEELHOUSE}"/cuopt*.whl \
    "${CUOPT_SH_CLIENT_WHEELHOUSE}"/cuopt_sh_client*.whl \
    "${LIBCUOPT_WHEELHOUSE}"/libcuopt*.whl

./datasets/linear_programming/download_pdlp_test_dataset.sh
./datasets/mip/download_miplib_test_dataset.sh

RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

EXITCODE=0
FAILED_STEPS=()
trap "EXITCODE=1" ERR
set +e

timeout 30m ./ci/run_cuopt_server_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-wheel-cuopt-server.xml" \
  --verbose --capture=no || FAILED_STEPS+=("pytest cuopt-server (wheel)")

# Run documentation tests
./ci/test_doc_examples.sh || FAILED_STEPS+=("doc examples")

# Generate nightly test report
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/utils/nightly_report_helper.sh"
generate_nightly_report "wheel-server" --with-python-version

if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    EXITCODE=1
    echo ""
    echo "==================== FAILED TEST STEPS (${#FAILED_STEPS[@]}) ===================="
    for s in "${FAILED_STEPS[@]}"; do echo "  - ${s}"; done
    echo "================================================================"
fi

exit ${EXITCODE}
