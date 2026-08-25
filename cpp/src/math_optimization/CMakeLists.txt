# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

list(PREPEND
  MATH_OPT_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_settings_gpu.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solution_reader.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solution_writer.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tic_toc.cpp
  )

# solver_settings_t is host-only apart from the device members in solver_settings_gpu.cu,
# so the bulk of it builds into cuopt_client. The gRPC client takes a solver_settings_t
# in its public API, so this is required for the client library to resolve standalone.
set(MATH_OPT_CLIENT_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_settings.cpp
  )

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES}
  ${MATH_OPT_SRC_FILES} PARENT_SCOPE)
set(CUOPT_CLIENT_SRC_FILES ${CUOPT_CLIENT_SRC_FILES}
  ${MATH_OPT_CLIENT_SRC_FILES} PARENT_SCOPE)
