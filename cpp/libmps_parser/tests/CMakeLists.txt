# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

function(ConfigureTest CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})

    target_compile_options(${CMAKE_TEST_NAME}
        PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${MPS_PARSER_CXX_FLAGS}>"
    )

    set_target_properties(${CMAKE_TEST_NAME}
        PROPERTIES
        # set target compile options
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
        CXX_SCAN_FOR_MODULES OFF
    )

    target_include_directories(${CMAKE_TEST_NAME}
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
        mps_parser
        GTest::gmock
        GTest::gmock_main
        GTest::gtest
        GTest::gtest_main
        ${CUOPT_PRIVATE_CUDA_LIBS}
    )

    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})

    install(
        TARGETS ${CMAKE_TEST_NAME}
        COMPONENT testing
        DESTINATION bin/gtests/libcuopt
        EXCLUDE_FROM_ALL
    )
endfunction()

###################################################################################################
# - Linear programming tests ----------------------------------------------------------------------
 ConfigureTest(MPS_PARSER_TEST
     mps_parser_test.cpp
 )
###################################################################################################
