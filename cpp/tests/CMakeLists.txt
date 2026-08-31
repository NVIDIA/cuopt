# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

enable_testing()

include(rapids-test)
rapids_test_init()

if(BUILD_TESTS)
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
      "${dejavu_SOURCE_DIR}"
  )

  target_link_libraries(cuopttestutils
      PUBLIC
      cuopt
      OpenMP::OpenMP_CXX
      OpenMP::OpenMP_CUDA
      GTest::gmock
      GTest::gtest
  )
  if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
    target_link_options(cuopttestutils PRIVATE -Wl,--enable-new-dtags)
  endif()

  # Static-linking flavor of the test utilities for tests built with STATIC_LIB
  # (see ConfigureTest). These link cuopt_static to reach internal symbols, so
  # their utilities must also resolve against cuopt_static rather than the shared
  # cuopt, which would otherwise mix static and shared cuopt in one binary.
  add_library(cuopttestutils_static STATIC
      routing/utilities/check_constraints.cu
  )

  target_compile_options(cuopttestutils_static
      PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUOPT_CXX_FLAGS}>"
      "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUOPT_CUDA_FLAGS}>>"
  )

  target_include_directories(cuopttestutils_static
      PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/../src"
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${dejavu_SOURCE_DIR}"
  )

  target_link_libraries(cuopttestutils_static
      PUBLIC
      cuopt_static
      OpenMP::OpenMP_CXX
      OpenMP::OpenMP_CUDA
      GTest::gmock
      GTest::gtest
  )
  if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
    target_link_options(cuopttestutils_static PRIVATE -Wl,--enable-new-dtags)
  endif()
endif()


set(CUOPT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ################################################################ ------------------------------------------------------------------
# ConfigureTest(NAME source1.cu source2.cu [STATIC_LIB] [LABELS label1 label2 ...])
#
# LABELS sets CTest labels for selective local test execution via `ctest -L <label>`.
# STATIC_LIB links cuopt_static instead of cuopt (for tests needing internal symbols).
function(ConfigureTest CMAKE_TEST_NAME)
    cmake_parse_arguments(CT "STATIC_LIB" "" "LABELS" ${ARGN})
    if(CT_STATIC_LIB)
        set(_cuopt_link_target cuopt_static)
        set(_extra_libs cuopttestutils_static)
    else()
        set(_cuopt_link_target cuopt)
        set(_extra_libs cuopttestutils)
    endif()
    add_executable(${CMAKE_TEST_NAME} ${CT_UNPARSED_ARGUMENTS})
    target_include_directories(${CMAKE_TEST_NAME}
        PRIVATE
        "${CUOPT_TEST_DIR}/../src"
        "${CUOPT_TEST_DIR}/../src/io"
        "${CUOPT_TEST_DIR}"
        "${papilo_SOURCE_DIR}/src"
        "${papilo_BINARY_DIR}"
        "${pslp_SOURCE_DIR}/include"
        "${dejavu_SOURCE_DIR}"
    )
    target_link_libraries(${CMAKE_TEST_NAME}
       PRIVATE
        ${_cuopt_link_target}
        ${_extra_libs}
        GTest::gmock
        GTest::gmock_main
        GTest::gtest
        GTest::gtest_main
        ${CUOPT_PRIVATE_CUDA_LIBS}
    )
    if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
      target_link_options(${CMAKE_TEST_NAME} PRIVATE -Wl,--enable-new-dtags)
    endif()

    rapids_test_add(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME} GPUS 1 INSTALL_COMPONENT_SET testing)

    if(CT_LABELS)
        set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES LABELS "${CT_LABELS}")
    endif()
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
if(BUILD_TESTS)
  if(NOT SKIP_ROUTING_BUILD)
    add_subdirectory(routing)
    add_subdirectory(distance_engine)
    add_subdirectory(examples)
  endif()
  if(NOT BUILD_LP_ONLY)
    add_subdirectory(mip)
  endif()
  add_subdirectory(linear_programming)
  add_subdirectory(internal)
  add_subdirectory(qp)
  add_subdirectory(utilities)

  rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcuopt)
endif()
