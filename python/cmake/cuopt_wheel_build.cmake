# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Shared build for the libcuopt wheels. Every wheel configures the same C++ tree and
# differs only in which install components scikit-build-core stages into it, set by
# install.components in each pyproject.toml. Keeping this in one file means the RPATH
# list and the third-party lookups cannot drift between the per-component wheels.

include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v3.2
)
FetchContent_MakeAvailable(argparse)

# gRPC must be available as an installed CMake package (gRPCConfig.cmake).
# On RockyLinux 8 wheel builds we install it in CI via ci/utils/install_protobuf_grpc.sh.
find_package(gRPC CONFIG REQUIRED)

find_package(Boost 1.65 REQUIRED)
if(Boost_FOUND)
    message(STATUS "Found Boost ${Boost_VERSION} in ${Boost_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Boost not found. Please install boost-devel.")
endif()

include_directories(${Boost_INCLUDE_DIRS})

set(BUILD_TESTING OFF CACHE BOOL "Disable test build for papilo")
set(PAPILO_NO_BINARIES ON)
option(LUSOL "Disable LUSOL" OFF)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(CUOPT_BUILD_TESTUTIL OFF)

add_subdirectory(../../cpp cuopt-cpp)

# cuopt is an INTERFACE target (libcuopt.so is a linker script) and compiles nothing,
# so it has no use for argparse. cuopt_cli and cuopt_grpc_server, which do, already link
# argparse::argparse in cpp/CMakeLists.txt.
target_link_libraries(cuopt_cli PRIVATE
    argparse
)

set(rpaths
  "$ORIGIN/../lib64"
  # The component libraries live in sibling wheels now, so a binary or library in one
  # package resolves the others through their install directories rather than its own.
  "$ORIGIN/../../libcuopt_client/lib64"
  "$ORIGIN/../../libcuopt_mathopt/lib64"
  "$ORIGIN/../../libcuopt_routing/lib64"
  "$ORIGIN/../../rapids_logger/lib64"
  "$ORIGIN/../../librmm/lib64"
  "$ORIGIN/../../nvidia/cudss/lib"
  "$ORIGIN/../../nvidia/cublas/lib"
  "$ORIGIN/../../nvidia/curand/lib"
  "$ORIGIN/../../nvidia/cusolver/lib"
  "$ORIGIN/../../nvidia/cusparse/lib"
  "$ORIGIN/../../nvidia/nccl/lib"
  "$ORIGIN/../../nvidia/nvjitlink/lib"
)

# Add CUDA version-specific paths based on CUDA compiler version
message(STATUS "libcuopt: CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 14.0)
  message(STATUS "libcuopt: Adding cu13 RPATH")
  list(APPEND rpaths "$ORIGIN/../../nvidia/cu13/lib")
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  message(STATUS "libcuopt: Adding cu12 RPATH")
  list(APPEND rpaths "$ORIGIN/../../nvidia/cu12/lib")
else()
  message(WARNING "libcuopt: Unsupported CUDA version ${CMAKE_CUDA_COMPILER_VERSION}")
endif()
message(STATUS "libcuopt: Final RPATH = ${rpaths}")

# cuopt is an INTERFACE target (libcuopt.so is a linker script), so it has no RPATH.
# cuopt_client is in the list because it PUBLIC-links rapids_logger, which
# build_wheel_libcuopt.sh excludes from vendoring, so the only way to find it is
# $ORIGIN/../../rapids_logger/lib64 from this list.
foreach(_target cuopt_routing cuopt_mathopt cuopt_client cuopt_cli cuopt_grpc_server)
  if(TARGET ${_target})
    set_property(TARGET ${_target} APPEND PROPERTY INSTALL_RPATH ${rpaths})
  endif()
endforeach()
