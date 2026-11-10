# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Included by linear_programming/CMakeLists.txt after ConfigureTest is defined.
# These generated-model CPU tests intentionally do not link cuopt: the probe
# test provides the registry definition once and checks access from another TU.
find_package(Threads REQUIRED)

function(ConfigurePdlpHostTest TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_compile_features(${TEST_NAME} PRIVATE cxx_std_20)
    target_include_directories(${TEST_NAME} PRIVATE "${CUOPT_TEST_DIR}/../src")
    target_link_libraries(${TEST_NAME} PRIVATE Threads::Threads)
    rapids_test_add(NAME ${TEST_NAME} COMMAND ${TEST_NAME} INSTALL_COMPONENT_SET testing)
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "numopt")
    install(
        TARGETS ${TEST_NAME}
        COMPONENT testing
        DESTINATION bin/gtests/libcuopt
        EXCLUDE_FROM_ALL
    )
endfunction()

ConfigurePdlpHostTest(PDLP_EQUIVALENT_PROBE_HOST_TEST
    ${CMAKE_CURRENT_LIST_DIR}/host_tests/equivalent_probe_test.cpp
    ${CMAKE_CURRENT_LIST_DIR}/host_tests/equivalent_probe_worker.cpp)
ConfigurePdlpHostTest(PDLP_RCM_ALGORITHM_HOST_TEST
    ${CMAKE_CURRENT_LIST_DIR}/host_tests/rcm_algorithm_test.cpp)
ConfigurePdlpHostTest(PDLP_NATIVE_RCM_HOST_TEST
    ${CMAKE_CURRENT_LIST_DIR}/host_tests/native_rcm_test.cpp)

# Static linkage also resolves the logger used by capture-invalidation recovery.
ConfigureTest(PDLP_OVERHEAD_TEST
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/reflected_projection_direct_test.cu
    ${CMAKE_CURRENT_LIST_DIR}/unit_tests/batched_halpern_test.cu
    LABELS numopt
    STATIC_LIB)
