# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(cython_sources internals.pyx)
# internals resolves no cuOpt symbols at all -- it is callback plumbing -- but still
# needs the public headers, which arrive through the target interface. The client is the
# leaf, so this gives headers without pulling in an engine.
set(linked_libraries cuopt::client)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX internals_
  ASSOCIATED_TARGETS cuopt
)

target_include_directories(internals_internals PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
