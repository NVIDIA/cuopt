# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(libcuopt-routing-python)

project(
  libcuopt-routing-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Already installed: nothing to build, the wheel just stages the component. Checking
# cuopt::routing specifically, not cuopt_FOUND, matters now that this build installs the
# client sibling wheel as a dependency -- that alone resolves cuopt_FOUND but not routing.
find_package(cuopt "${RAPIDS_VERSION}")
if(TARGET cuopt::routing)
  return()
endif()
unset(cuopt_FOUND)

include(../cmake/cuopt_wheel_build.cmake)
