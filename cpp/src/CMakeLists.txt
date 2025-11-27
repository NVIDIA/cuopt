# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(UTIL_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utilities/seed_generator.cu
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/logger.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/version_info.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/utilities/timestamp_utils.cpp)

add_subdirectory(linear_programming)
add_subdirectory(math_optimization)
add_subdirectory(mip)

# Only build routing for full builds, not LP-only builds
if(NOT SKIP_ROUTING_BUILD)
  add_subdirectory(routing)
endif()

add_subdirectory(dual_simplex)

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES} ${UTIL_SRC_FILES} PARENT_SCOPE)
