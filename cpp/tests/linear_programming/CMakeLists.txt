# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

ConfigureTest(LP_UNIT_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/optimization_problem_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/solver_settings_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/presolve_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/solution_interface_test.cu
)# ##################################################################################################
# - Linear programming PDLP tests ----------------------------------------------------------------------
ConfigureTest(PDLP_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/pdlp_test.cu
)

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
        "${CUOPT_TEST_DIR}/../libmps_parser/src"
        "${CUOPT_TEST_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/c_api_tests"
    )

    target_link_libraries(C_API_TEST
        PRIVATE
        mps_parser
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

    install(
        TARGETS C_API_TEST
        COMPONENT testing
        DESTINATION bin/gtests/libcuopt
        EXCLUDE_FROM_ALL
    )
endif()
