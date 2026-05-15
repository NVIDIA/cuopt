# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(CUOPT)

project(
  cuopt-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUOPT_PYTHON_COMPONENT "FULL" CACHE STRING "Python package slice: FULL, LP (math), or ROUTING")
set_property(CACHE CUOPT_PYTHON_COMPONENT PROPERTY STRINGS FULL LP ROUTING)

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cuopt/_build_variant.py.in"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuopt/_build_variant.py"
        @ONLY
)

find_package(cuopt "${RAPIDS_VERSION}")
if(NOT CUOPT_PYTHON_COMPONENT STREQUAL "ROUTING")
  find_package(mps_parser "${RAPIDS_VERSION}")
endif()

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(cuopt)
if(NOT CUOPT_PYTHON_COMPONENT STREQUAL "ROUTING")
  add_subdirectory(cuopt/linear_programming/internals/)
endif()
