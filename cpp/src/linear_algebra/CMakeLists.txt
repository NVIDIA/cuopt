# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(LINEAR_ALGEBRA_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_matrix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/vector_math.cpp
  )

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES}
  ${LINEAR_ALGEBRA_SRC_FILES} PARENT_SCOPE)
