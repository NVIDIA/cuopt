# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Level-0 routing tests need internal (hidden) symbols, so they link cuopt_static.
# They are consolidated into a single static binary rather than one static binary
# per test: each STATIC_LIB target embeds the full multi-arch cuopt_static device
# code, so N standalone static binaries would explode the libcuopt-tests package
# size. See cpp/tests/internal/CMakeLists.txt for the same consolidation pattern.
ConfigureTest(ROUTING_INTERNAL_TEST
    ${CUOPT_TEST_DIR}/internal/main.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_ges_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_routing_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_vehicle_order_match.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_vehicle_types_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_objective_function_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/top_k.cu
    LABELS routing
    STATIC_LIB)

# Level-1 routing regression tests, disabled by default. Combined into a single
# static binary so the two never-run tests don't ship as two separate multi-arch
# cuopt_static binaries.
#
# The target name intentionally ends in "L1TEST", NOT "_TEST": ci/run_ctests.sh
# runs every installed binary matching "*_TEST", and these regression tests are
# not maintained to pass in CI. Ending the name outside that glob keeps them out
# of the CI run (matching the original RETAIL_L1TEST/ROUTING_L1TEST names). The
# DISABLED property below only affects local `ctest` invocations. Do not rename
# this to end in "_TEST" without also excluding it from run_ctests.sh.
ConfigureTest(ROUTING_L1TEST
    ${CUOPT_TEST_DIR}/internal/main.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/level1/l1_retail_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/level1/l1_routing_test.cu
    LABELS routing
    STATIC_LIB)
set_tests_properties(ROUTING_L1TEST PROPERTIES DISABLED TRUE)

# # - ${CMAKE_CURRENT_SOURCE_DIR} unit tests ----------------------------------------------------------------------------
ConfigureTest(ROUTING_UNIT_TEST
      ${CUOPT_TEST_DIR}/internal/main.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/vehicle_types.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/breaks.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/heterogenous_breaks.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/vehicle_fixed_costs.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/vehicle_order_match.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/order_locations.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/horizontal_loading.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/route_constraints.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/capacity_route_resize.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/heterogenous_fleet.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/prize_collection.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/objective_function.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/batch_tsp.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/set_shmem_of_kernel.cu
    LABELS routing
    STATIC_LIB)
