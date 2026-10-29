# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Files necessary for Linear Programming functionality
set(PRESOLVE_LP_NECESSARY_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party_presolve.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/single_lock_dual_aggregation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gf2_presolve.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bhw_coeff_reduce.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/indicator_strengthening.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conflict_graph/clique_table.cu
)

# Files that are MIP-specific and not needed for pure LP
set(PRESOLVE_NON_LP_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/block_bve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/bounds_presolve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/bounds_update_data.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/semi_continuous.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/conditional_bound_strengthening.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/multi_probe.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/probing_cache.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/trivial_presolve.cu
)

if(BUILD_LP_ONLY)
  set(PRESOLVE_SRC_FILES ${PRESOLVE_LP_NECESSARY_FILES})
else()
  set(PRESOLVE_SRC_FILES ${PRESOLVE_LP_NECESSARY_FILES} ${PRESOLVE_NON_LP_FILES})
endif()

set(PRESOLVE_SRC_FILES ${PRESOLVE_SRC_FILES} PARENT_SCOPE)
set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES}
  ${PRESOLVE_SRC_FILES} PARENT_SCOPE)
