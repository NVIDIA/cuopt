# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

ConfigureTest(LP_UNIT_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/solver_settings_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/solution_interface_test.cu
    LABELS numopt)
# ##################################################################################################
# - Linear programming PDLP tests ----------------------------------------------------------------------
# Single-GPU pdlp_test.cu is compiled into NUMOPT_INTERNAL_TEST
# (cpp/tests/internal/CMakeLists.txt). Under hidden symbol visibility it needs
# cuopt_static, so it runs there rather than as a separate shared-linked binary.

# Multi-GPU distributed PDLP tests. Binary name ends in _MG_TEST so that
# ci/test_cpp_multi_gpu.sh picks it up via its *_MG_TEST glob. Each test
# calls GTEST_SKIP when fewer than 2 GPUs are visible so the binary is
# harmless on single-GPU runners. STATIC_LIB: needs internal (hidden) PDLP symbols.
ConfigureTest(PDLP_MG_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/pdlp_distributed_test.cu
    LABELS numopt
    STATIC_LIB)

# ##################################################################################################
# - MPS / LP parser tests --------------------------------------------------------------------------
# parser_test.cpp is compiled into NUMOPT_INTERNAL_TEST (static); it needs internal
# symbols hidden from the shared lib, so it runs there rather than standalone.
ConfigureTest(MPS_FAST_PARSER_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/experimental_mps_fast/fast_fp64_parser_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/experimental_mps_fast/fast_parser_edge_test.cpp
    LABELS numopt
    STATIC_LIB)
target_include_directories(MPS_FAST_PARSER_TEST
    PRIVATE
    "${CUOPT_TEST_DIR}/../src/io/experimental_mps_fast"
)
target_link_libraries(MPS_FAST_PARSER_TEST PRIVATE simde::simde)


# ##################################################################################################
# - C API Tests----------------------------------------------------------------------
# The C API tests require a separate library to be linked against. So we don't use the ConfigureTest macro.

if (NOT SKIP_C_PYTHON_ADAPTERS)
    add_library(c_api_tester STATIC "${CMAKE_CURRENT_SOURCE_DIR}/c_api_tests/c_api_test.c")
    target_include_directories(c_api_tester
        PRIVATE
        ${CUOPT_SOURCE_DIR}/include
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
    )
    target_include_directories(c_api_tester
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/c_api_tests"
    )

    target_link_libraries(c_api_tester
        PRIVATE
        CUDA::cudart
    )

    add_executable(C_API_TEST "${CMAKE_CURRENT_SOURCE_DIR}/c_api_tests/c_api_tests.cpp")

    target_include_directories(C_API_TEST
        PRIVATE
        "${CUOPT_TEST_DIR}/../src"
        "${CUOPT_TEST_DIR}/../src/io"
        "${CUOPT_TEST_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/c_api_tests"
    )

    target_link_libraries(C_API_TEST
        PRIVATE
        cuopt
        cuopttestutils
        c_api_tester
        GTest::gmock
        GTest::gmock_main
        GTest::gtest
        GTest::gtest_main
        ${CUOPT_PRIVATE_CUDA_LIBS}
    )

    if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
      target_link_options(C_API_TEST PRIVATE -Wl,--enable-new-dtags)
    endif()

    add_test(NAME C_API_TEST COMMAND C_API_TEST)
    set_tests_properties(C_API_TEST PROPERTIES LABELS "numopt")

    install(
        TARGETS C_API_TEST
        COMPONENT testing
        DESTINATION bin/gtests/libcuopt
        EXCLUDE_FROM_ALL
    )
endif()

# ##################################################################################################
# - gRPC Tests -------------------------------------------------------------------------------------
if(CUOPT_ENABLE_GRPC)
    add_subdirectory(grpc)
endif()
