#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Toolchain for the static Java build inside a RAPIDS ci-wheel container. Source, do not execute.
#
# ci-conda's gcc_linux-64 conda package produces a libstdc++ ABI newer than the RAPIDS-supported
# platforms (Ubuntu 22.04, Rocky Linux 8) ship, so a JNI library built with it fails to load for
# real consumers -- exactly the bug cuDF hit in production (rapidsai/cudf#23753, fixed in
# rapidsai/cudf#23766 by rebuilding on ci-wheel with Red Hat's gcc-toolset instead, which is built
# for this "modern compiler, older runtime" use case). This mirrors that fix: build on ci-wheel
# (Rocky Linux 8) with gcc-toolset-14.
#
# There is no conda environment here at all, so every C++ build dependency has to come from
# somewhere else. Rather than inventing a new way to source them, this reuses exactly the sources
# ci/build_wheel_libcuopt.sh already relies on for the same conda-free ci-wheel environment:
# dnf (via ci/utils/*.sh) for Boost/TBB/cuDSS, the base image itself for NCCL (already present via
# NVIDIA's own cuda dnf repo), and prebuilt RAPIDS wheels for rmm/rapids_logger (raft has no wheel,
# so rapids-cmake's usual CPM source-fetch fallback still applies to it, exactly as it would in a
# conda environment lacking a prebuilt raft package).
#
# Installs/activates a JDK, ninja, cmake, gcc-toolset-14, and Boost/TBB/cuDSS/rmm/rapids_logger.
# Safe to source repeatedly (JAVA_STATIC_ENV_READY short-circuits after the first load).

TOOLSET_VERSION="${TOOLSET_VERSION:-14}"

if [[ -n "${JAVA_STATIC_ENV_READY:-}" ]]; then
  return 0
fi

# shellcheck disable=SC1091
source rapids-init-pip

rapids-logger "Installing JDK, Maven, ninja and cmake"
# dnf's own maven and cmake packages are too old for this project's needs; get ninja/cmake from
# PyPI, matching how the build side already gets modern tooling in this family of images, and
# Maven from Apache's own binaries, matching ci/test_java_static.sh's identical installation.
dnf install -y java-17-openjdk-devel
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
rapids-pip-retry install cmake ninja

MAVEN_VERSION="3.9.16"
MAVEN_HOME="$(mktemp -d)"
MAVEN_TARBALL="$(mktemp)"
# dlcdn.apache.org (not archive.apache.org) is Apache's actual mirror-network-backed download
# endpoint -- archive.apache.org is documented as permanent, unmirrored storage for old releases,
# not meant for routine/automated downloads, and has been observed intermittently unreachable
# (connection timeouts) from CI runners as a result. dlcdn only serves the current release per
# branch, so MAVEN_VERSION has to track that (currently 3.9.16, the latest 3.9.x). Matches
# ci/test_java_static.sh's identical download.
MAVEN_TARBALL_URL="https://dlcdn.apache.org/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz"
# Retry on top of the mirror switch: even a mirror-backed CDN can have a transient blip, and
# retrying is cheap insurance against failing an otherwise-passing job over it.
CURL_RETRY_ARGS=(--retry 5 --retry-delay 5 --retry-connrefused)
curl -fsSL "${CURL_RETRY_ARGS[@]}" "${MAVEN_TARBALL_URL}" -o "${MAVEN_TARBALL}"
# Verify against Apache's published SHA-512 rather than trusting transport security alone.
echo "$(curl -fsSL "${CURL_RETRY_ARGS[@]}" "${MAVEN_TARBALL_URL}.sha512")  ${MAVEN_TARBALL}" | sha512sum --check --status
tar xz -C "${MAVEN_HOME}" --strip-components=1 -f "${MAVEN_TARBALL}"
rm -f "${MAVEN_TARBALL}"
export PATH="${MAVEN_HOME}/bin:${PATH}"

rapids-logger "Activating gcc-toolset-${TOOLSET_VERSION}"
# shellcheck disable=SC1090,SC1091
. "/opt/rh/gcc-toolset-${TOOLSET_VERSION}/enable"
export CC="/opt/rh/gcc-toolset-${TOOLSET_VERSION}/root/usr/bin/gcc"
export CXX="/opt/rh/gcc-toolset-${TOOLSET_VERSION}/root/usr/bin/g++"
export CUDAHOSTCXX="${CXX}"

rapids-logger "Installing rockylinux repo, Boost, TBB and cuDSS"
bash ci/utils/update_rockylinux_repo.sh
bash ci/utils/install_boost_tbb.sh
# cpp/CMakeLists.txt never links Papilo's own CMake target (only its source/build directories,
# to avoid confusing clang's include resolution -- see its "Adding Papilo as a system include"
# comment), so Papilo's own #include <boost/...> lines are resolved purely through the
# compiler's default system include search, not anything CMake adds. A conda environment's
# boost-cpp package satisfies that for free by installing unversioned headers straight under
# $CONDA_PREFIX/include, which conda's own compiler activation already puts on that search path.
# EPEL's boost1.78-devel instead installs versioned headers under /usr/include/boost1.78, which
# is on no default search path at all, so it has to be added explicitly. CPATH is the portable,
# standard way to extend a compiler's default include search (as opposed to a project-specific
# -I flag), which is the right level here since nothing in this repo's CMake is boost1.78-aware.
export CPATH="/usr/include/boost1.78${CPATH:+:${CPATH}}"
# install_cudss.sh keys off CUDA_VERSION (unlike the rest of this repo's RAPIDS_CUDA_VERSION),
# matching ci/build_wheel_libcuopt.sh's own usage.
CUDA_VERSION="${CUDA_VERSION:-${RAPIDS_CUDA_VERSION}}" bash ci/utils/install_cudss.sh
# NCCL is intentionally not installed here: libnccl-devel already ships in the base ci-wheel
# CUDA-devel image via NVIDIA's own cuda dnf repo.

rapids-logger "Installing rmm and rapids_logger from prebuilt RAPIDS wheels"
# libraft-headers has no wheel; depends_on_libraft_headers is conda-only, so this generates a
# requirements file with only rmm/rapids_logger in it, and CMake's usual CPM source-fetch fallback
# (see cpp/CMakeLists.txt) picks up raft from GitHub instead, same as it would with no conda
# environment providing a prebuilt raft package.
rapids-dependency-file-generator \
  --output requirements \
  --file-key py_build_libcuopt \
  --file-key py_rapids_build_libcuopt \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-java-static.txt
rapids-pip-retry install -v --prefer-binary -r /tmp/requirements-java-static.txt

# rmm/rapids_logger install their CMake package config files under their own site-packages
# directory rather than a location find_package() searches by default; point CMake at them so it
# resolves the prebuilt wheels instead of falling through to CPM's source-fetch fallback.
CUOPT_JAVA_STATIC_CMAKE_PREFIX_PATH="$(
  python3 -c '
import importlib.util, sys
paths = []
for mod in ("librmm", "rapids_logger"):
    spec = importlib.util.find_spec(mod)
    if spec and spec.submodule_search_locations:
        paths.append(next(iter(spec.submodule_search_locations)))
print(";".join(paths))
'
)"
export CUOPT_JAVA_STATIC_CMAKE_PREFIX_PATH

# cuDSS's prebuilt threading-layer plugin (libcudss_mtlayer_gomp.so.0) needs libgomp's OpenMP 5.0
# detached-task support (omp_fulfill_event) at runtime; Rocky 8's own libgomp -- including
# gcc-toolset-14's own copy -- doesn't have it (confirmed: neither exports the symbol). Fetching a
# modern one from conda-forge and building our own threading-layer plugin against it instead
# (CUOPT_BUILD_CUSTOM_CUDSS_MTLAYER, set by build_static_libcuopt.sh) sidesteps that entirely,
# mirroring ci/build_wheel_libcuopt.sh's identical fix for the same underlying problem (#1219,
# #1905): a missing symbol can't be worked around by better library resolution, only a genuinely
# newer libgomp fixes it.
rapids-logger "Fetching a modern libgomp for cuDSS's threading layer"
CUOPT_MODERN_LIBGOMP_DIR="${PWD}/modern_libgomp"
rapids-pip-retry install zstandard
python3 ci/utils/install_modern_libgomp.py "${CUOPT_MODERN_LIBGOMP_DIR}"
export CUOPT_MODERN_LIBGOMP_DIR
export LD_LIBRARY_PATH="${CUOPT_MODERN_LIBGOMP_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export JAVA_STATIC_ENV_READY=1
