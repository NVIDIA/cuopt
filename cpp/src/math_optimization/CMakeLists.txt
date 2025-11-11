# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

list(PREPEND
  MATH_OPT_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_settings.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solution_reader.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solution_writer.cu
  )

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES}
  ${MATH_OPT_SRC_FILES} PARENT_SCOPE)
