# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(libcuopt-python)

project(
  libcuopt-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Debug: Print CUDA compiler version early
message(STATUS "=============================================================")
message(STATUS "libcuopt: CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "libcuopt: CUDAToolkit_VERSION = ${CUDAToolkit_VERSION}")
message(STATUS "=============================================================")

# Check if cuopt is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuopt package.
find_package(cuopt "${RAPIDS_VERSION}")

if(cuopt_FOUND)
  return()
endif()

unset(cuopt_FOUND)

include(../cmake/cuopt_wheel_build.cmake)
