# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(libcuopt-mathopt-python)

project(
  libcuopt-mathopt-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Already installed: nothing to build, the wheel just stages the component.
find_package(cuopt "${RAPIDS_VERSION}")
if(cuopt_FOUND)
  return()
endif()
unset(cuopt_FOUND)

include(../cmake/cuopt_wheel_build.cmake)
