#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Install Protobuf and gRPC C++ development libraries from source.
#
# This script builds gRPC, Protobuf, and Abseil from source to ensure consistent
# ABI and avoid symbol issues (notably abseil-cpp#1624: Mutex::Dtor not exported
# from shared libabseil on Linux).
#
# Usage:
#   ./install_protobuf_grpc.sh [OPTIONS]
#
# Options:
#   --prefix=DIR       Installation prefix (default: /usr/local)
#   --build-dir=DIR    Build directory for source builds (default: /tmp)
#   --skip-deps        Skip installing system dependencies (for conda builds)
#   --help             Show this help message
#
# Examples:
#   # Wheel builds (install to /usr/local, installs system deps)
#   ./install_protobuf_grpc.sh
#
#   # Conda builds (install to custom prefix, deps already available)
#   ./install_protobuf_grpc.sh --prefix=${GRPC_INSTALL_DIR} --build-dir=${SRC_DIR} --skip-deps

# Configuration - single source of truth for gRPC version
GRPC_VERSION="v1.64.2"

# Default values
PREFIX="/usr/local"
BUILD_DIR="/tmp"
SKIP_DEPS=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix=*)
            PREFIX="${1#*=}"
            shift
            ;;
        --build-dir=*)
            BUILD_DIR="${1#*=}"
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Build and install gRPC ${GRPC_VERSION} and dependencies from source."
            echo ""
            echo "Options:"
            echo "  --prefix=DIR       Installation prefix (default: /usr/local)"
            echo "  --build-dir=DIR    Build directory for source builds (default: /tmp)"
            echo "  --skip-deps        Skip installing system dependencies (for conda builds)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PREFIX=$(realpath -m "$PREFIX" 2>/dev/null || readlink -f "$PREFIX" 2>/dev/null || echo "$PREFIX")
BUILD_DIR=$(realpath -m "$BUILD_DIR" 2>/dev/null || readlink -f "$BUILD_DIR" 2>/dev/null || echo "$BUILD_DIR")

if [[ -z "$PREFIX" || "$PREFIX" == "/" ]]; then
    echo "ERROR: Invalid PREFIX: '$PREFIX'" >&2
    exit 1
fi
if [[ -z "$BUILD_DIR" || "$BUILD_DIR" == "/" ]]; then
    echo "ERROR: Invalid BUILD_DIR: '$BUILD_DIR'" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

echo "=============================================="
echo "Installing gRPC ${GRPC_VERSION} from source"
echo "  Prefix: ${PREFIX}"
echo "  Build dir: ${BUILD_DIR}"
echo "  Skip deps: ${SKIP_DEPS}"
echo "=============================================="

# Install system dependencies if not skipped
if [ "${SKIP_DEPS}" = false ]; then
    echo ""
    echo "Installing system dependencies..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" == "rocky" || "$ID" == "centos" || "$ID" == "rhel" || "$ID" == "fedora" ]]; then
            # Enable PowerTools (Rocky 8) or CRB (Rocky 9) for some packages
            if [[ "${VERSION_ID%%.*}" == "8" ]]; then
                dnf config-manager --set-enabled powertools || dnf config-manager --set-enabled PowerTools || true
            elif [[ "${VERSION_ID%%.*}" == "9" ]]; then
                dnf config-manager --set-enabled crb || true
            fi
            dnf install -y git cmake ninja-build gcc gcc-c++ openssl-devel zlib-devel c-ares-devel
        elif [[ "$ID" == "ubuntu" || "$ID" == "debian" ]]; then
            apt-get update
            apt-get install -y git cmake ninja-build g++ libssl-dev zlib1g-dev libc-ares-dev
        else
            echo "Warning: Unknown OS '$ID'. Assuming build tools are already installed."
        fi
    else
        echo "Warning: /etc/os-release not found. Assuming build tools are already installed."
    fi
fi

# Verify required tools are available
echo ""
echo "Checking required tools..."
for tool in git cmake ninja; do
    if ! command -v "$tool" &> /dev/null; then
        echo "Error: Required tool '$tool' not found. Please install it (e.g., via your package manager) and re-run this script."
        exit 1
    fi
done
echo "All required tools found."

# Clean up any previous installations to avoid ABI mismatches
# (notably Abseil LTS namespaces like absl::lts_20220623 vs absl::lts_20250512)
echo "Cleaning up previous installations..."
rm -rf \
  "${PREFIX}/lib/cmake/grpc" "${PREFIX}/lib64/cmake/grpc" \
  "${PREFIX}/lib/cmake/protobuf" "${PREFIX}/lib64/cmake/protobuf" \
  "${PREFIX}/lib/cmake/absl" "${PREFIX}/lib64/cmake/absl" \
  "${PREFIX}/include/absl" "${PREFIX}/include/google/protobuf" "${PREFIX}/include/grpc" \
  "${PREFIX}/bin/grpc_cpp_plugin" "${PREFIX}/bin/protoc" "${PREFIX}/bin/protoc-"* || true
rm -f \
  "${PREFIX}/lib/"libgrpc*.a "${PREFIX}/lib/"libgpr*.a "${PREFIX}/lib/"libaddress_sorting*.a "${PREFIX}/lib/"libre2*.a "${PREFIX}/lib/"libupb*.a \
  "${PREFIX}/lib64/"libabsl_*.a "${PREFIX}/lib64/"libprotobuf*.so* "${PREFIX}/lib64/"libprotoc*.so* \
  "${PREFIX}/lib/"libprotobuf*.a "${PREFIX}/lib/"libprotoc*.a || true

# Build and install gRPC dependencies from source in a consistent way.
#
# IMPORTANT: Protobuf and gRPC both depend on Abseil, and the Abseil LTS
# namespace (e.g. absl::lts_20250512) is part of C++ symbol mangling.
# If Protobuf and gRPC are built against different Abseil versions, gRPC
# plugins can fail to link with undefined references (e.g. Printer::PrintImpl).
#
# To avoid that, we install Abseil first (from gRPC's submodule), then
# build Protobuf and gRPC against that same installed Abseil.

GRPC_SRC="${BUILD_DIR}/grpc-src"
ABSL_BUILD="${BUILD_DIR}/absl-build"
PROTOBUF_BUILD="${BUILD_DIR}/protobuf-build"
GRPC_BUILD="${BUILD_DIR}/grpc-build"

rm -rf "${GRPC_SRC}" "${ABSL_BUILD}" "${PROTOBUF_BUILD}" "${GRPC_BUILD}"
mkdir -p "${PREFIX}"

echo "Cloning gRPC ${GRPC_VERSION} with submodules..."
git clone --depth 1 --branch "${GRPC_VERSION}" --recurse-submodules --shallow-submodules \
    https://github.com/grpc/grpc.git "${GRPC_SRC}"

# Ensure prefix is in PATH and CMAKE_PREFIX_PATH
export PATH="${PREFIX}/bin:${PATH}"
export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"

# Ensure a consistent C++ standard across Abseil/Protobuf/gRPC.
# Abseil's options.h defaults to "auto" selection for std::string_view
# (ABSL_OPTION_USE_STD_STRING_VIEW=2). If one library is built in
# C++17+ and another in C++14, they will disagree on whether
# `absl::string_view` is a typedef to `std::string_view` or Abseil's
# own type, leading to link-time ABI mismatches.
CMAKE_STD_FLAGS=("-DCMAKE_CXX_STANDARD=17" "-DCMAKE_CXX_STANDARD_REQUIRED=ON")

echo ""
echo "Building Abseil (from gRPC submodule)..."
cmake -S "${GRPC_SRC}/third_party/abseil-cpp" -B "${ABSL_BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    "${CMAKE_STD_FLAGS[@]}" \
    -DABSL_PROPAGATE_CXX_STD=ON \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"
cmake --build "${ABSL_BUILD}" --parallel
cmake --install "${ABSL_BUILD}"

echo ""
echo "Building Protobuf (using installed Abseil)..."
cmake -S "${GRPC_SRC}/third_party/protobuf" -B "${PROTOBUF_BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    "${CMAKE_STD_FLAGS[@]}" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_ABSL_PROVIDER=package \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"
cmake --build "${PROTOBUF_BUILD}" --parallel
cmake --install "${PROTOBUF_BUILD}"

echo ""
echo "Building gRPC (using installed Abseil and Protobuf)..."
cmake -S "${GRPC_SRC}" -B "${GRPC_BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    "${CMAKE_STD_FLAGS[@]}" \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DgRPC_BUILD_CODEGEN=ON \
    -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
    -DgRPC_ABSL_PROVIDER=package \
    -DgRPC_PROTOBUF_PROVIDER=package \
    -DgRPC_RE2_PROVIDER=module \
    -DgRPC_SSL_PROVIDER=package \
    -DgRPC_ZLIB_PROVIDER=package \
    -DgRPC_CARES_PROVIDER=package \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"
cmake --build "${GRPC_BUILD}" --parallel
cmake --install "${GRPC_BUILD}"

# For system-wide installs, update ldconfig
if [[ "${PREFIX}" == "/usr/local" ]]; then
    echo ""
    echo "Updating ldconfig for system-wide install..."
    echo "${PREFIX}/lib64" > /etc/ld.so.conf.d/usr-local-lib64.conf 2>/dev/null || true
    echo "${PREFIX}/lib" > /etc/ld.so.conf.d/usr-local-lib.conf 2>/dev/null || true
    ldconfig || true
fi

echo ""
echo "=============================================="
echo "gRPC ${GRPC_VERSION} installed successfully to ${PREFIX}"
echo "=============================================="
