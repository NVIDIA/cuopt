# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# One extension module for both arms. See grpc_client.pyx for why they share a
# unit: they wrap the same C++ client object, and keeping them together lets the
# client be detached from the engines as a single package later.
set(cython_sources grpc_client.pyx)
# rmm is named here rather than inherited: cuopt_client carries no rmm dependency (#1890),
# but Cython emits catch handlers for rmm's exception types, so this extension needs them.
set(linked_libraries cuopt::client rmm::rmm)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
  MODULE_PREFIX grpc_client_
  ASSOCIATED_TARGETS cuopt
)
