# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(cython_sources grpc_client.pyx)
set(linked_libraries cuopt::cuopt)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
  MODULE_PREFIX grpc_
  ASSOCIATED_TARGETS cuopt
)

foreach(_cython_target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  target_include_directories(
    ${_cython_target} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../linear_programming/solver")
endforeach()
