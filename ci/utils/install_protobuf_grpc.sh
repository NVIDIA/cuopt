#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Install Protobuf and gRPC C++ development libraries.
# On RockyLinux 8, grpc-devel is often unavailable; in that case we build and
# install gRPC (and a modern Protobuf) from source so CMake can find:
# - gRPCConfig.cmake
# - grpc_cpp_plugin
# - protoc
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "rocky" ]]; then
        echo "Detected Rocky Linux. Installing Protobuf + gRPC via dnf..."
        # Enable PowerTools (Rocky 8) or CRB (Rocky 9) repository for protobuf-devel
        if [[ "${VERSION_ID%%.*}" == "8" ]]; then
            dnf config-manager --set-enabled powertools || dnf config-manager --set-enabled PowerTools || true
        elif [[ "${VERSION_ID%%.*}" == "9" ]]; then
            dnf config-manager --set-enabled crb || true
        fi
        dnf install -y git curl ca-certificates

        # Base build deps for source builds (safe even if grpc-devel exists)
        #dnf install -y \
        #    cmake ninja-build make gcc gcc-c++ \
        #    openssl-devel zlib-devel c-ares-devel
        dnf install -y ninja-build openssl-devel zlib-devel c-ares-devel

        # Protobuf from distro (keeps protoc available even if source build is skipped)
        # dnf install -y protobuf-devel protobuf-compiler

        # Try system gRPC first (may not exist on Rocky 8 images).
        if dnf install -y grpc-devel grpc-plugins; then
            echo "Installed gRPC from dnf."
        elif dnf install -y grpc-devel; then
            echo "Installed grpc-devel from dnf (grpc_cpp_plugin may be missing)."
        else
            echo "grpc-devel not available. Building and installing Protobuf + gRPC from source..."

            PREFIX="/usr/local"

            # If the container previously had a different source-installed gRPC/Protobuf/Abseil
            # under /usr/local, wipe those bits to avoid ABI mismatches (notably Abseil LTS
            # namespaces like absl::lts_20220623 vs absl::lts_20250512).
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

            # Keep in sync with conda (grpc-cpp) baseline.
            GRPC_VERSION="v1.51.1"
            rm -rf /tmp/grpc-src /tmp/grpc-build /tmp/absl-build
            git clone --depth 1 --branch "${GRPC_VERSION}" --recurse-submodules https://github.com/grpc/grpc.git /tmp/grpc-src

            # Ensure /usr/local is preferred for tools/libs
            export PATH="${PREFIX}/bin:${PATH}"
            export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"

            # Ensure a consistent C++ standard across Abseil/Protobuf/gRPC.
            # Abseil's options.h defaults to "auto" selection for std::string_view
            # (ABSL_OPTION_USE_STD_STRING_VIEW=2). If one library is built in
            # C++17+ and another in C++14, they will disagree on whether
            # `absl::string_view` is a typedef to `std::string_view` or Abseil's
            # own type, leading to link-time ABI mismatches.
            CMAKE_STD_FLAGS="-DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON"

            echo "Building and installing Abseil (from gRPC submodule) into ${PREFIX}..."
            cmake -S /tmp/grpc-src/third_party/abseil-cpp -B /tmp/absl-build -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
                ${CMAKE_STD_FLAGS} \
                -DABSL_PROPAGATE_CXX_STD=ON \
                -DCMAKE_INSTALL_PREFIX="${PREFIX}"
            cmake --build /tmp/absl-build
            cmake --install /tmp/absl-build

            echo "Building and installing Protobuf into ${PREFIX} (using installed Abseil)..."
            # Match a Protobuf version known to work with gRPC v1.51.x.
            PROTOBUF_VERSION="v3.21.12"
            rm -rf /tmp/protobuf-src /tmp/protobuf-build
            git clone --depth 1 --branch "${PROTOBUF_VERSION}" https://github.com/protocolbuffers/protobuf.git /tmp/protobuf-src
            cmake -S /tmp/protobuf-src -B /tmp/protobuf-build -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
                ${CMAKE_STD_FLAGS} \
                -Dprotobuf_BUILD_TESTS=OFF \
                -DBUILD_SHARED_LIBS=ON \
                -Dprotobuf_ABSL_PROVIDER=package \
                -DCMAKE_INSTALL_PREFIX="${PREFIX}"
            cmake --build /tmp/protobuf-build
            cmake --install /tmp/protobuf-build

            # Build and install gRPC against the installed Protobuf + Abseil and system c-ares.
            # We only need the C++ codegen plugin for cuOpt.

            cmake -S /tmp/grpc-src -B /tmp/grpc-build -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
                ${CMAKE_STD_FLAGS} \
                -DgRPC_INSTALL=ON \
                -DgRPC_BUILD_TESTS=OFF \
                -DgRPC_BUILD_CODEGEN=ON \
                -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
                -DgRPC_SSL_PROVIDER=package \
                -DgRPC_ZLIB_PROVIDER=package \
                -DgRPC_CARES_PROVIDER=package \
                -DgRPC_PROTOBUF_PROVIDER=package \
                -DgRPC_ABSL_PROVIDER=package \
                -DgRPC_RE2_PROVIDER=module \
                -DCMAKE_INSTALL_PREFIX="${PREFIX}"
            cmake --build /tmp/grpc-build
            cmake --install /tmp/grpc-build

            # Avoid accidentally mixing Rocky's old protobuf headers/libs (3.5.x)
            # with the source-installed protobuf (31.x) during subsequent builds.
            # dnf remove -y protobuf-compiler protobuf-devel protobuf || true

            # Ensure the runtime linker can find /usr/local libs (Rocky8 doesn't
            # always include /usr/local/lib64 in its default search paths).
            echo "${PREFIX}/lib64" > /etc/ld.so.conf.d/usr-local-lib64.conf
            echo "${PREFIX}/lib" > /etc/ld.so.conf.d/usr-local-lib.conf
            ldconfig || true
        fi
    elif [[ "$ID" == "ubuntu" ]]; then
        echo "Detected Ubuntu. Installing Protobuf + gRPC via apt..."
        apt-get update
        # Protobuf (headers + protoc)
        apt-get install -y libprotobuf-dev protobuf-compiler

        # gRPC C++ (headers/libs + grpc_cpp_plugin for codegen)
        apt-get install -y libgrpc++-dev protobuf-compiler-grpc
    else
        echo "Unknown OS: $ID. Please install Protobuf + gRPC development libraries manually."
        exit 1
    fi
else
    echo "/etc/os-release not found. Cannot determine OS. Please install Protobuf + gRPC development libraries manually."
    exit 1
fi
