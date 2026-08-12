# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(UTIL_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utilities/seed_generator.cu
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/logger.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/version_info.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/timestamp_utils.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/work_unit_scheduler.cpp)

add_subdirectory(linear_algebra)
add_subdirectory(pdlp)
add_subdirectory(math_optimization)
add_subdirectory(mip_heuristics)
add_subdirectory(io)

# Only build routing for full builds, not LP-only builds
if(NOT SKIP_ROUTING_BUILD)
  add_subdirectory(routing)
endif()

add_subdirectory(dual_simplex)
add_subdirectory(barrier)
add_subdirectory(branch_and_bound)
add_subdirectory(cuts)

# Aggregate per-domain source lists for the three component libraries
set(CUOPT_BASE_SRC_FILES
  ${UTIL_SRC_FILES}
)

set(CUOPT_ROUTING_SRC_FILES
  ${ROUTING_SRC_FILES}
)

set(CUOPT_MATHOPT_SRC_FILES
  ${LINEAR_ALGEBRA_SRC_FILES}
  ${LP_SRC_FILES}
  ${MATH_OPT_SRC_FILES}
  ${MIP_SRC_FILES}
  ${PARSERS_SRC_FILES}
  ${DUAL_SIMPLEX_SRC_FILES}
  ${BARRIER_SRC_FILES}
  ${BRANCH_AND_BOUND_SRC_FILES}
  ${CUTS_SRC_FILES}
)

set(CUOPT_BASE_SRC_FILES ${CUOPT_BASE_SRC_FILES} PARENT_SCOPE)
set(CUOPT_ROUTING_SRC_FILES ${CUOPT_ROUTING_SRC_FILES} PARENT_SCOPE)
set(CUOPT_MATHOPT_SRC_FILES ${CUOPT_MATHOPT_SRC_FILES} PARENT_SCOPE)
set(CUOPT_SRC_FILES
  ${CUOPT_BASE_SRC_FILES}
  ${CUOPT_ROUTING_SRC_FILES}
  ${CUOPT_MATHOPT_SRC_FILES}
  PARENT_SCOPE
)
set(MPS_FAST_SRC_FILES ${MPS_FAST_SRC_FILES} PARENT_SCOPE)
