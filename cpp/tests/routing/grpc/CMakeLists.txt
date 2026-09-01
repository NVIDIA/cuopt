# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Standalone VRP gRPC test driver (not a gtest). Requires a running cuopt_grpc_server.

include(${CUOPT_SOURCE_DIR}/cmake/thirdparty/get_nlohmann_json.cmake)

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

target_link_libraries(GRPC_VRP_TEST_DRIVER
    PRIVATE
    cuopt
    gRPC::grpc++
    protobuf::libprotobuf
    nlohmann_json::nlohmann_json
)

if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
  target_link_options(GRPC_VRP_TEST_DRIVER PRIVATE -Wl,--enable-new-dtags)
endif()

add_dependencies(GRPC_VRP_TEST_DRIVER cuopt_grpc_server)

# Mapper round-trip (no running server). Covers time-window and distance
# per-vehicle breaks with the same proto grouping (vehicle_id + repeated breaks).
add_executable(GRPC_ROUTING_PROBLEM_MAPPER_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/grpc_routing_problem_mapper_test.cpp
)

target_include_directories(GRPC_ROUTING_PROBLEM_MAPPER_TEST
    PRIVATE
    "${CUOPT_SOURCE_DIR}/include"
    "${CUOPT_SOURCE_DIR}/src/grpc"
    "${CUOPT_SOURCE_DIR}/src/grpc/codegen/generated"
    "${CMAKE_BINARY_DIR}"
)

target_link_libraries(GRPC_ROUTING_PROBLEM_MAPPER_TEST
    PRIVATE
    cuopt
    GTest::gtest
    GTest::gtest_main
    protobuf::libprotobuf
)

if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
  target_link_options(GRPC_ROUTING_PROBLEM_MAPPER_TEST PRIVATE -Wl,--enable-new-dtags)
endif()

add_test(NAME GRPC_ROUTING_PROBLEM_MAPPER_TEST COMMAND GRPC_ROUTING_PROBLEM_MAPPER_TEST)
set_tests_properties(GRPC_ROUTING_PROBLEM_MAPPER_TEST PROPERTIES LABELS "routing")

install(
    TARGETS GRPC_ROUTING_PROBLEM_MAPPER_TEST
    COMPONENT testing
    DESTINATION bin/gtests/libcuopt
    EXCLUDE_FROM_ALL
)

install(
    TARGETS GRPC_VRP_TEST_DRIVER
    COMPONENT testing
    DESTINATION bin/gtests/libcuopt
    EXCLUDE_FROM_ALL
)
