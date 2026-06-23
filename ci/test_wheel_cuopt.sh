#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# sets up a constraints file for 'pip' and puts its location in an exported variable PIP_EXPORT,
# so those constraints will affect all future 'pip install' calls
source rapids-init-pip

# Make sure libssl.so.3 / libcrypto.so.3 are on the linker path before any
# 'import cuopt'. cuopt wheels link OpenSSL 3 and don't bundle it; on Rocky 8
# the runtime needs to come from EPEL (no-op on distros that already ship it).
bash "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/utils/install_openssl3_runtime.sh"

# Download the packages built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBCUOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
CUOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuopt cuopt --py "$RAPIDS_PY_VERSION" --cuda "$RAPIDS_CUDA_VERSION")")
CUOPT_SH_CLIENT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuopt-sh-client cuopt --pure --arch any)")

# update pip constraints.txt to ensure all future 'pip install' (including those in ci/thirdparty-testing)
# use these wheels for cuopt packages
cat > "${PIP_CONSTRAINT}" <<EOF
cuopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${CUOPT_WHEELHOUSE}"/cuopt_"${RAPIDS_PY_CUDA_SUFFIX}"-*.whl)
cuopt-sh-client @ file://$(echo "${CUOPT_SH_CLIENT_WHEELHOUSE}"/cuopt_sh_client-*.whl)
libcuopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUOPT_WHEELHOUSE}"/libcuopt_"${RAPIDS_PY_CUDA_SUFFIX}"-*.whl)
EOF

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
    --constraint "${PIP_CONSTRAINT}" \
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

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
export RAPIDS_TESTS_DIR
mkdir -p "${RAPIDS_TESTS_DIR}"

EXITCODE=0
FAILED_STEPS=()
trap "EXITCODE=1" ERR
set +e

# Run CLI tests
timeout 10m bash ./python/libcuopt/libcuopt/tests/test_cli.sh || FAILED_STEPS+=("cuopt_cli")

# Run Python tests
timeout 30m ./ci/run_cuopt_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-wheel-cuopt.xml" \
  --verbose --capture=no || FAILED_STEPS+=("pytest cuopt (wheel)")

# run thirdparty integration tests for only nightly builds
if [[ "${RAPIDS_BUILD_TYPE}" == "nightly" ]]; then
    ./ci/thirdparty-testing/run_jump_tests.sh || FAILED_STEPS+=("thirdparty jump")
    ./ci/thirdparty-testing/run_cvxpy_tests.sh || FAILED_STEPS+=("thirdparty cvxpy")
    ./ci/thirdparty-testing/run_pulp_tests.sh || FAILED_STEPS+=("thirdparty pulp")
    ./ci/thirdparty-testing/run_pyomo_tests.sh || FAILED_STEPS+=("thirdparty pyomo")
fi

# Generate nightly test report
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/utils/nightly_report_helper.sh"
generate_nightly_report "wheel-python" --with-python-version

if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    EXITCODE=1
    echo ""
    echo "==================== FAILED TEST STEPS (${#FAILED_STEPS[@]}) ===================="
    for s in "${FAILED_STEPS[@]}"; do echo "  - ${s}"; done
    echo "================================================================"
fi

exit ${EXITCODE}
