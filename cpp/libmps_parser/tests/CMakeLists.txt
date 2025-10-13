# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        INTERFACE_POSITION_INDEPENDENT_CODE ON
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
