# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(UTIL_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utilities/seed_generator.cu
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/version_info.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/timestamp_utils.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/work_unit_scheduler.cpp)

# No sources: <utilities/logger.hpp> became header-only in #1778.
set(UTIL_CLIENT_SRC_FILES)

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

# Aggregate per-domain source lists for the component libraries.
# UTIL_SRC_FILES lives in cuopt_mathopt rather than a separate cuopt_base: mathopt is its
# only consumer (routing, grpc and client reference none of its symbols), so a standalone
# library would exist to serve one caller.
set(CUOPT_ROUTING_SRC_FILES
  ${ROUTING_SRC_FILES}
)

# PARSERS_SRC_FILES is deliberately absent here: the parsers are CUDA-free and build into
# cuopt_client, which cuopt_mathopt links. Listing them in both would compile them twice.
set(CUOPT_MATHOPT_SRC_FILES
  ${UTIL_SRC_FILES}
  ${LINEAR_ALGEBRA_SRC_FILES}
  ${LP_SRC_FILES}
  ${MATH_OPT_SRC_FILES}
  ${MIP_SRC_FILES}
  ${DUAL_SIMPLEX_SRC_FILES}
  ${BARRIER_SRC_FILES}
  ${BRANCH_AND_BOUND_SRC_FILES}
  ${CUTS_SRC_FILES}
)

set(CUOPT_ROUTING_SRC_FILES ${CUOPT_ROUTING_SRC_FILES} PARENT_SCOPE)
set(CUOPT_MATHOPT_SRC_FILES ${CUOPT_MATHOPT_SRC_FILES} PARENT_SCOPE)
set(CUOPT_CLIENT_SRC_FILES ${CUOPT_CLIENT_SRC_FILES} ${UTIL_CLIENT_SRC_FILES} PARENT_SCOPE)
set(CUOPT_SRC_FILES
  ${CUOPT_ROUTING_SRC_FILES}
  ${CUOPT_MATHOPT_SRC_FILES}
  PARENT_SCOPE
)
set(MPS_FAST_SRC_FILES ${MPS_FAST_SRC_FILES} PARENT_SCOPE)
