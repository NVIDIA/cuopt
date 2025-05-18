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

set(cuopt_version 25.05.00)

include(../../cmake/rapids_config.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(libcuopt-python)

project(
  libcuopt-python
  VERSION "${cuopt_version}"
  LANGUAGES CXX CUDA
)

# Check if cuopt is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuopt package.
find_package(cuopt "${cuopt_version}")

if(cuopt_FOUND)
  return()
endif()

unset(cuopt_FOUND)


include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(CUOPT_BUILD_TESTUTIL OFF)
set(CUDA_STATIC_RUNTIME ON)

add_subdirectory(../../cpp cuopt-cpp)

target_link_libraries(cuopt PRIVATE argparse)
target_link_libraries(cuopt_cli PRIVATE argparse)

set(rpaths
  "$ORIGIN/../lib64"
  "$ORIGIN/../../nvidia/cublas/lib"
  "$ORIGIN/../../nvidia/curand/lib"
  "$ORIGIN/../../nvidia/cusolver/lib"
  "$ORIGIN/../../nvidia/cusparse/lib"
)

set_property(TARGET cuopt PROPERTY INSTALL_RPATH ${rpaths} APPEND)
set_property(TARGET cuopt_cli PROPERTY INSTALL_RPATH ${rpaths} APPEND)
