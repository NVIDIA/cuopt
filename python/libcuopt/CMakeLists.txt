# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(libcuopt-python)

project(
  libcuopt-python
  VERSION "${CUOPT_VERSION}"
  LANGUAGES CXX CUDA
)

# Debug: Print CUDA compiler version early
message(STATUS "=============================================================")
message(STATUS "libcuopt: CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "libcuopt: CUDAToolkit_VERSION = ${CUDAToolkit_VERSION}")
message(STATUS "=============================================================")

# Check if cuopt is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuopt package.
find_package(cuopt "${CUOPT_VERSION}")

if(cuopt_FOUND)
  return()
endif()

unset(cuopt_FOUND)

include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v3.2
)
FetchContent_MakeAvailable(argparse)


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
set(CUDA_STATIC_RUNTIME ON)

add_subdirectory(../../cpp cuopt-cpp)

target_link_libraries(cuopt PRIVATE
    argparse
)
target_link_libraries(cuopt_cli PRIVATE
    argparse
)

set(rpaths
  "$ORIGIN/../lib64"
  "$ORIGIN/../../rapids_logger/lib64"
  "$ORIGIN/../../librmm/lib64"
  "$ORIGIN/../../nvidia/cudss/lib"
  "$ORIGIN/../../nvidia/cublas/lib"
  "$ORIGIN/../../nvidia/curand/lib"
  "$ORIGIN/../../nvidia/cusolver/lib"
  "$ORIGIN/../../nvidia/cusparse/lib"
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

set_property(TARGET cuopt PROPERTY INSTALL_RPATH ${rpaths} APPEND)
set_property(TARGET cuopt_cli PROPERTY INSTALL_RPATH ${rpaths} APPEND)
