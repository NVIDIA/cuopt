# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(CUTS_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/clique.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cut_pool.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cuts.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flow_cover.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gomory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/implied_bound.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/knapsack.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mixed_integer_rounding.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/objective_step.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/strong_cg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/variable_bounds.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/zero_half.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/zero_half_mod2.cpp
  )

set(CUTS_SRC_FILES ${CUTS_SRC_FILES} PARENT_SCOPE)
set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES}
  ${CUTS_SRC_FILES} PARENT_SCOPE)
