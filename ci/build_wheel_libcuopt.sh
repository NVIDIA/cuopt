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

# Install libuuid (needed by cuopt_grpc_server)
if command -v dnf &> /dev/null; then
    dnf install -y libuuid-devel
elif command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y uuid-dev
fi

# Install Protobuf + gRPC (protoc + grpc_cpp_plugin)
bash ci/utils/install_protobuf_grpc.sh

# EXPERIMENT (#1219): use GNU libgomp instead of bundling LLVM libomp, to test whether the
# constraint that motivated the LLVM switch in #1429 still holds. It doesn't fully: GNU
# libgomp has implemented the OpenMP 5.0 detached-task runtime (omp_fulfill_event) that the
# fast MPS parser needs since GCC 9, but the *default* '-lgomp' resolution on the Rocky Linux
# 8 wheel build image picks up the base OS's pre-5.0 libgomp rather than the SCL gcc-toolset
# compiler's own newer one. Resolve a libgomp.so that actually exports the symbol we need,
# the same way the (now-removed) LLVM libomp override resolved a concrete versioned ELF file
# instead of trusting default discovery.
LIBGOMP_LIBRARY="$(
    { ldconfig -p | awk '$1 ~ /^libgomp\.so(\.[0-9]+)*$/ { print $NF }'
      find /opt/rh /usr -name 'libgomp.so.1*' 2>/dev/null
    } | sort -u | while read -r candidate; do
        nm -D "${candidate}" 2>/dev/null | grep -qE ' T omp_fulfill_event(@|$)' && echo "${candidate}" && break
    done
)" || true
if [[ "${LIBGOMP_LIBRARY}" != /* || ! -f "${LIBGOMP_LIBRARY}" ]]; then
    echo "Could not resolve a GNU libgomp with OpenMP 5.0 detached-task support (omp_fulfill_event)" >&2
    exit 1
fi

echo "Using GNU libgomp runtime: ${LIBGOMP_LIBRARY}"

export SKBUILD_CMAKE_ARGS="-DOpenMP_gomp_LIBRARY:FILEPATH=${LIBGOMP_LIBRARY}"

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
  --exclude "libnccl.so.*"
  --exclude "libnvJitLink.so*"
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
  # EXPERIMENT (#1219): don't bundle/hash-rename libgomp. Resolving it from the host at
  # runtime instead keeps cuOpt on the same libgomp.so.1 instance that cuDSS's threading
  # layer (libcudss_mtlayer_gomp.so.0) loads, instead of two independent OpenMP runtimes.
  --exclude "libgomp.so.*"
)

ci/build_wheel.sh libcuopt ${package_dir}

mkdir -p final_dist
python -m auditwheel repair "${EXCLUDE_ARGS[@]}" -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" ${package_dir}/dist/*

ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
