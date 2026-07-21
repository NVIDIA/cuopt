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
  MODULE_PREFIX grpc_routing_
  ASSOCIATED_TARGETS cuopt
)
