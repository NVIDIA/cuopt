# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# All routing tests include routing_test.cuh which pulls in cuopt::routing::detail
# types (solution_t, problem_t, fleet_order_constraints_t, md_utils, etc.).
# They must link against cuopt_static to access those hidden symbols.
ConfigureInternalTest(ROUTING_TEST ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_routing_test.cu
    LABELS routing)
ConfigureInternalTest(ROUTING_GES_TEST ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_ges_test.cu
    LABELS routing)

ConfigureInternalTest(VEHICLE_ORDER_TEST ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_vehicle_order_match.cu
    LABELS routing)
ConfigureInternalTest(VEHICLE_TYPES_TEST ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_vehicle_types_test.cu
    LABELS routing)
ConfigureInternalTest(OBJECTIVE_FUNCTION_TEST ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_objective_function_test.cu
    LABELS routing)

# ##################################################################################################
# - L1 advanced retail tests --------------------------------------------------------------------------
ConfigureInternalTest(RETAIL_L1TEST ${CMAKE_CURRENT_SOURCE_DIR}/level1/l1_retail_test.cu
    LABELS routing)

# ##################################################################################################
# - L1  tests for quick regression check --------------------------------------------------------------------------
ConfigureInternalTest(ROUTING_L1TEST ${CMAKE_CURRENT_SOURCE_DIR}/level1/l1_routing_test.cu
    LABELS routing)

# Disabled: both tests are slow and known to be broken.
set_tests_properties(RETAIL_L1TEST ROUTING_L1TEST PROPERTIES DISABLED TRUE)

# # - ${CMAKE_CURRENT_SOURCE_DIR} unit tests ----------------------------------------------------------------------------
ConfigureInternalTest(ROUTING_UNIT_TEST
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/vehicle_types.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/breaks.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/heterogenous_breaks.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/vehicle_fixed_costs.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/vehicle_order_match.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/order_locations.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/horizontal_loading.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/route_constraints.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/heterogenous_fleet.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/prize_collection.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/objective_function.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/top_k.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/batch_tsp.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/set_shmem_of_kernel.cu
    LABELS routing)
