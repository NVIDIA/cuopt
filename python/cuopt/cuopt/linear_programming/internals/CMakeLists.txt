# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(cython_sources internals.pyx)
set(linked_libraries cuopt::cuopt)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX internals_
  ASSOCIATED_TARGETS cuopt
)

target_include_directories(internals_internals PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
