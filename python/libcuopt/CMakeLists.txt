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

# Already installed: nothing to build, the wheel just stages the component. Checking for
# the cuopt_grpc_server binary itself, not cuopt_FOUND, matters now that this build installs
# the client/mathopt/routing sibling wheels as dependencies -- those alone resolve cuopt_FOUND
# but not a prebuilt cuopt_grpc_server. PATHS+NO_DEFAULT_PATH restricts the search to the same
# install trees find_package(cuopt) resolved, not an unrelated binary elsewhere on PATH.
find_package(cuopt "${RAPIDS_VERSION}")
find_program(CUOPT_GRPC_SERVER_PREBUILT cuopt_grpc_server
    PATHS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
)

if(cuopt_FOUND AND CUOPT_GRPC_SERVER_PREBUILT)
  return()
endif()

unset(cuopt_FOUND)

include(../cmake/cuopt_wheel_build.cmake)
