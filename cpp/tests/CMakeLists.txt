# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

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
  )

  target_link_libraries(cuopttestutils
      PUBLIC
      cuopt
      GTest::gmock
      GTest::gtest
  )
  if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
    target_link_options(cuopttestutils PRIVATE -Wl,--enable-new-dtags)
  endif()

  # Global property to accumulate test-name:label mappings for the installed manifest.
  define_property(GLOBAL PROPERTY CUOPT_TEST_LABEL_ENTRIES
      BRIEF_DOCS "Lines for test_labels.txt (NAME:label1,label2)")
  set_property(GLOBAL PROPERTY CUOPT_TEST_LABEL_ENTRIES "")
endif()


set(CUOPT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Helper: append a test's labels to the global manifest property.
function(register_test_labels TEST_NAME)
    list(JOIN ARGN "," _labels)
    set_property(GLOBAL APPEND PROPERTY CUOPT_TEST_LABEL_ENTRIES "${TEST_NAME}:${_labels}")
endfunction()

# ################################################################ ------------------------------------------------------------------
# ConfigureTest(NAME source1.cu source2.cu [LABELS label1 label2 ...])
#
# LABELS sets CTest labels (for local `ctest -L <label>`) and registers the
# test in the installed test_labels.txt manifest (for CI label filtering).
function(ConfigureTest CMAKE_TEST_NAME)
    cmake_parse_arguments(CT "" "" "LABELS" ${ARGN})
    add_executable(${CMAKE_TEST_NAME} ${CT_UNPARSED_ARGUMENTS})
    set_target_properties(${CMAKE_TEST_NAME}
      PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(${CMAKE_TEST_NAME}
        PRIVATE
        "${CUOPT_TEST_DIR}/../src"
        "${CUOPT_TEST_DIR}/../libmps_parser/src"
        "${CUOPT_TEST_DIR}"
        "${papilo_SOURCE_DIR}/src"
        "${papilo_BINARY_DIR}"
        "${pslp_SOURCE_DIR}/include"
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
    if(NOT DEFINED INSTALL_TARGET OR "${INSTALL_TARGET}" STREQUAL "")
      target_link_options(${CMAKE_TEST_NAME} PRIVATE -Wl,--enable-new-dtags)
    endif()

    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})

    if(CT_LABELS)
        set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES LABELS "${CT_LABELS}")
        register_test_labels(${CMAKE_TEST_NAME} ${CT_LABELS})
    endif()

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
  add_subdirectory(dual_simplex)
  add_subdirectory(qp)
  add_subdirectory(utilities)

  enable_testing()

  # Register the MPS_PARSER_TEST labels (defined in libmps_parser/tests/,
  # which has its own ConfigureTest that does not call register_test_labels).
  register_test_labels(MPS_PARSER_TEST mathopt parser)

  # Generate and install the test-label manifest so ci/run_ctests.sh can
  # filter tests by label without needing a CMake build directory.
  get_property(_entries GLOBAL PROPERTY CUOPT_TEST_LABEL_ENTRIES)
  list(JOIN _entries "\n" _manifest_content)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test_labels.txt" "${_manifest_content}\n")
  install(
      FILES "${CMAKE_CURRENT_BINARY_DIR}/test_labels.txt"
      COMPONENT testing
      DESTINATION bin/gtests/libcuopt
      EXCLUDE_FROM_ALL
  )
endif()
