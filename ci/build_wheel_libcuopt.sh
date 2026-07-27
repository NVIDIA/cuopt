#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_name="libcuopt"
package_dir="python/libcuopt"

# Install rockylinux repo
if command -v dnf &> /dev/null; then
    bash ci/utils/update_rockylinux_repo.sh
fi

# Install Boost and TBB
bash ci/utils/install_boost_tbb.sh

# Install libuuid and LLVM's OpenMP runtime
if command -v dnf &> /dev/null; then
    # LLVM Toolset is distributed as a module on Rocky/RHEL 8.
    dnf module install -y llvm-toolset
    dnf install -y libuuid-devel libomp-devel
elif command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y uuid-dev libomp-dev
fi

# Install Protobuf + gRPC (protoc + grpc_cpp_plugin)
bash ci/utils/install_protobuf_grpc.sh

# Compile with GCC, but use LLVM libomp as the OpenMP runtime bundled in the wheel. Resolve the
# versioned ELF library rather than an unversioned linker script or compiler-toolset indirection.
LIBOMP_LIBRARY="$(
    ldconfig -p |
        awk '$1 ~ /^libomp\.so(\.[0-9]+)*$/ && !library { library = $NF }
             END { print library }'
)"
if [[ "${LIBOMP_LIBRARY}" != /* || ! -f "${LIBOMP_LIBRARY}" ]]; then
    echo "Could not resolve the LLVM OpenMP runtime: '${LIBOMP_LIBRARY}'" >&2
    exit 1
fi

echo "Using LLVM OpenMP runtime: ${LIBOMP_LIBRARY}"

export SKBUILD_CMAKE_ARGS="-DOpenMP_gomp_LIBRARY:FILEPATH=${LIBOMP_LIBRARY}"

# OpenSSL 3 hints for libcuopt's own find_package(OpenSSL).
#
# install_protobuf_grpc.sh links gRPC against OpenSSL 3 (see that script for
# rationale). libcuopt then re-resolves OpenSSL via find_package because
# gRPC's imported targets propagate it transitively. On Rocky/RHEL 8 the
# EPEL openssl3-devel package installs in non-default paths, so we have to
# point CMake at them; on Rocky/RHEL 9+ and Ubuntu 22.04+ the default
# OpenSSL is already 3.x and no hints are needed.
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "rocky" || "$ID" == "centos" || "$ID" == "rhel" || "$ID" == "fedora" ]] && \
       [[ "${VERSION_ID%%.*}" == "8" ]]; then
        SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};-DOPENSSL_INCLUDE_DIR=/usr/include/openssl3;-DOPENSSL_SSL_LIBRARY=/usr/lib64/openssl3/libssl.so;-DOPENSSL_CRYPTO_LIBRARY=/usr/lib64/openssl3/libcrypto.so"
    fi
fi

# For pull requests we are enabling assert mode.
if [ "$RAPIDS_BUILD_TYPE" = "pull-request" ]; then
    echo "Building in assert mode"
    export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};-DDEFINE_ASSERT=True"
else
    echo "Building in release mode"
fi

# Install cudss
bash ci/utils/install_cudss.sh

rapids-logger "Generating build requirements"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0


EXCLUDE_ARGS=(
  --exclude "libraft.so"
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcuda.so.1"
  --exclude "libcudss.so.*"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libnvJitLink*"
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
  # OpenSSL 3 is intentionally NOT bundled. Resolving libssl.so.3 / libcrypto.so.3
  # at runtime via the host (or container image) keeps libcrypto and the FIPS
  # provider (system or mounted) byte-version-matched, which is required for
  # the FIPS provider's HMAC integrity check and avoids loading two libcrypto.so.3
  # in the same process. Hosts must provide libssl.so.3 / libcrypto.so.3 (Ubuntu
  # 22.04+, RHEL/Rocky 9+, manylinux_2_28+ with openssl3, Debian 12+).
  --exclude "libssl.so.3"
  --exclude "libcrypto.so.3"
)

ci/build_wheel.sh libcuopt ${package_dir}

mkdir -p final_dist
python -m auditwheel repair "${EXCLUDE_ARGS[@]}" -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" ${package_dir}/dist/*

ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
