# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# This directory will contain routing examples
# Add example executables here as they are created

# Add routing examples
add_executable(service_team_routing service_team_routing.cu)
add_executable(pdptw_mixed_fleet pdptw_mixed_fleet.cu)
add_executable(cvrp_daily_deliveries cvrp_daily_deliveries.cu)

# Set CUDA standard for all examples
set_target_properties(
  service_team_routing
  pdptw_mixed_fleet
  cvrp_daily_deliveries
  PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
)

# Link libraries for all examples
foreach(target
    service_team_routing
    pdptw_mixed_fleet
    cvrp_daily_deliveries
)
  target_include_directories(${target}
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../src"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../routing/utilities"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../src/utilities"
  )
  target_link_libraries(${target}
    PRIVATE
      cuopt
      cuopttestutils
      OpenMP::OpenMP_CXX
  )
endforeach()
