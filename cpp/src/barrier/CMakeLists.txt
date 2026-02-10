# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(BARRIER_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/cusparse_view.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/barrier.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/device_sparse_matrix.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/pinned_host_allocator.cu
  )

# Uncomment to enable debug info
#set_source_files_properties(${BARRIER_SRC_FILES} DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTIES COMPILE_OPTIONS "-g1")

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES}
  ${BARRIER_SRC_FILES} PARENT_SCOPE)
