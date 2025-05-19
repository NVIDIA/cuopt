# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_library(cuopttestutils STATIC
    routing/utilities/check_constraints.cu
)

target_compile_options(cuopttestutils
    PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUOPT_CXX_FLAGS}>"
    "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUOPT_CUDA_FLAGS}>>"
)

target_include_directories(cuopttestutils
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(cuopttestutils
    PUBLIC
    cuopt
    GTest::gmock
    GTest::gtest
)

set(CUOPT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ################################################################ ------------------------------------------------------------------
function(ConfigureTest CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})

    target_include_directories(${CMAKE_TEST_NAME}
        PRIVATE
        "${CUOPT_TEST_DIR}/../src"
        "${CUOPT_TEST_DIR}/../libmps_parser/src"
        "${CUOPT_TEST_DIR}"
    )

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
        mps_parser
        cuopt
        cuopttestutils
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


# ####################################################################
# - set rapids dataset path ----------------------------------------------------------------------
if(RAPIDS_DATASET_ROOT_DIR)
    message(STATUS "setting default RAPIDS_DATASET_ROOT_DIR to: ${RAPIDS_DATASET_ROOT_DIR}")
    string(CONCAT CMAKE_C_FLAGS ${CMAKE_C_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
    string(CONCAT CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
    string(CONCAT CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
endif(RAPIDS_DATASET_ROOT_DIR)

# ## test sources
add_subdirectory(routing)
add_subdirectory(linear_programming)
add_subdirectory(distance_engine)
add_subdirectory(mip)
add_subdirectory(dual_simplex)
add_subdirectory(examples)
add_subdirectory(utilities)

enable_testing()