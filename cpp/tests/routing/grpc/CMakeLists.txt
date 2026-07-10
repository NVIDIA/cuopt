# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Standalone VRP gRPC test driver (not a gtest). Requires a running cuopt_grpc_server.

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  # Fall back to system header if the CMake package is unavailable.
  find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
  if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(WARNING "nlohmann_json not found; skipping GRPC_VRP_TEST_DRIVER")
    return()
  endif()
endif()

add_executable(GRPC_VRP_TEST_DRIVER
    ${CMAKE_CURRENT_SOURCE_DIR}/grpc_vrp_test_driver.cpp
)

target_include_directories(GRPC_VRP_TEST_DRIVER
    PRIVATE
    "${CUOPT_SOURCE_DIR}/include"
    "${CUOPT_SOURCE_DIR}/src/grpc"
    "${CUOPT_SOURCE_DIR}/src/grpc/codegen/generated"
    "${CMAKE_BINARY_DIR}"
)

if(NLOHMANN_JSON_INCLUDE_DIR)
  target_include_directories(GRPC_VRP_TEST_DRIVER PRIVATE "${NLOHMANN_JSON_INCLUDE_DIR}")
endif()

target_link_libraries(GRPC_VRP_TEST_DRIVER
    PRIVATE
    cuopt
    gRPC::grpc++
    protobuf::libprotobuf
)

if(TARGET nlohmann_json::nlohmann_json)
  target_link_libraries(GRPC_VRP_TEST_DRIVER PRIVATE nlohmann_json::nlohmann_json)
endif()

if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
  target_link_options(GRPC_VRP_TEST_DRIVER PRIVATE -Wl,--enable-new-dtags)
endif()

add_dependencies(GRPC_VRP_TEST_DRIVER cuopt_grpc_server)

install(
    TARGETS GRPC_VRP_TEST_DRIVER
    COMPONENT testing
    DESTINATION bin/gtests/libcuopt
    EXCLUDE_FROM_ALL
)
