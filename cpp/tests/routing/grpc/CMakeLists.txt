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

install(
    TARGETS GRPC_VRP_TEST_DRIVER
    COMPONENT testing
    DESTINATION bin/gtests/libcuopt
    EXCLUDE_FROM_ALL
)
