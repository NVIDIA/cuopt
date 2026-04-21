# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# ##################################################################################################
# - gRPC Client Unit Tests -------------------------------------------------------------------------
# These tests use mock stubs and don't require a running server

add_executable(GRPC_CLIENT_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/grpc_client_test.cpp
)

target_include_directories(GRPC_CLIENT_TEST
    PRIVATE
    "${CUOPT_SOURCE_DIR}/include"
    "${CUOPT_SOURCE_DIR}/src/grpc"
    "${CUOPT_SOURCE_DIR}/src/grpc/client"
    "${CUOPT_TEST_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"  # For grpc_client_test_helper.hpp
    "${CMAKE_BINARY_DIR}"  # For generated protobuf headers
)

target_link_libraries(GRPC_CLIENT_TEST
    PRIVATE
    cuopt
    GTest::gmock
    GTest::gmock_main
    gRPC::grpc++
    protobuf::libprotobuf
)

if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
  target_link_options(GRPC_CLIENT_TEST PRIVATE -Wl,--enable-new-dtags)
endif()

add_test(NAME GRPC_CLIENT_TEST COMMAND GRPC_CLIENT_TEST)
set_tests_properties(GRPC_CLIENT_TEST PROPERTIES LABELS "numopt")

install(
    TARGETS GRPC_CLIENT_TEST
    COMPONENT testing
    DESTINATION bin/gtests/libcuopt
    EXCLUDE_FROM_ALL
)

# ##################################################################################################
# - gRPC Pipe Serialization Unit Tests -------------------------------------------------------------
# These tests verify pipe serialization round-trips using real pipe(2) FDs.
# No running server is required; pipe I/O functions are self-contained in the test.

add_executable(GRPC_PIPE_SERIALIZATION_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/grpc_pipe_serialization_test.cpp
    "${CUOPT_SOURCE_DIR}/src/grpc/server/grpc_pipe_io.cpp"
    "${CUOPT_SOURCE_DIR}/src/grpc/server/grpc_server_logger.cpp"
)

target_include_directories(GRPC_PIPE_SERIALIZATION_TEST
    PRIVATE
    "${CUOPT_SOURCE_DIR}/include"
    "${CUOPT_SOURCE_DIR}/src/grpc"
    "${CUOPT_SOURCE_DIR}/src/grpc/server"
    "${CMAKE_BINARY_DIR}"  # For generated protobuf headers
)

target_link_libraries(GRPC_PIPE_SERIALIZATION_TEST
    PRIVATE
    cuopt
    GTest::gtest
    GTest::gtest_main
    protobuf::libprotobuf
)

if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
  target_link_options(GRPC_PIPE_SERIALIZATION_TEST PRIVATE -Wl,--enable-new-dtags)
endif()

add_test(NAME GRPC_PIPE_SERIALIZATION_TEST COMMAND GRPC_PIPE_SERIALIZATION_TEST)
set_tests_properties(GRPC_PIPE_SERIALIZATION_TEST PROPERTIES LABELS "numopt")

install(
    TARGETS GRPC_PIPE_SERIALIZATION_TEST
    COMPONENT testing
    DESTINATION bin/gtests/libcuopt
    EXCLUDE_FROM_ALL
)

# ##################################################################################################
# - gRPC Integration Tests -------------------------------------------------------------------------
# These tests start a real server process and run end-to-end tests

add_executable(GRPC_INTEGRATION_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/grpc_integration_test.cpp
)

target_include_directories(GRPC_INTEGRATION_TEST
    PRIVATE
    "${CUOPT_SOURCE_DIR}/include"
    "${CUOPT_SOURCE_DIR}/src/grpc"
    "${CUOPT_SOURCE_DIR}/src/grpc/client"
    "${CUOPT_TEST_DIR}"
    "${CMAKE_BINARY_DIR}"  # For generated protobuf headers
)

target_link_libraries(GRPC_INTEGRATION_TEST
    PRIVATE
    cuopt
    GTest::gmock
    GTest::gmock_main
    gRPC::grpc++
    protobuf::libprotobuf
)

if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
  target_link_options(GRPC_INTEGRATION_TEST PRIVATE -Wl,--enable-new-dtags)
endif()

# Integration tests need the server binary to be built first
add_dependencies(GRPC_INTEGRATION_TEST cuopt_grpc_server)

add_test(
    NAME GRPC_INTEGRATION_TEST
    COMMAND ${CMAKE_COMMAND} -E env
        "CUOPT_GRPC_SERVER_PATH=$<TARGET_FILE:cuopt_grpc_server>"
        $<TARGET_FILE:GRPC_INTEGRATION_TEST>
)
set_tests_properties(GRPC_INTEGRATION_TEST PROPERTIES LABELS "numopt")

install(
    TARGETS GRPC_INTEGRATION_TEST
    COMPONENT testing
    DESTINATION bin/gtests/libcuopt
    EXCLUDE_FROM_ALL
)
