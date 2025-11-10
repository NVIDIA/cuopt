# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

ConfigureTest(ROUTING_TEST ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_routing_test.cu)
ConfigureTest(ROUTING_GES_TEST ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_ges_test.cu)

ConfigureTest(VEHICLE_ORDER_TEST ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_vehicle_order_match.cu)
ConfigureTest(VEHICLE_TYPES_TEST ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_vehicle_types_test.cu)
ConfigureTest(OBJECTIVE_FUNCTION_TEST ${CMAKE_CURRENT_SOURCE_DIR}/level0/l0_objective_function_test.cu)

# ##################################################################################################
# - L1 advanced retail tests --------------------------------------------------------------------------
ConfigureTest(RETAIL_L1TEST ${CMAKE_CURRENT_SOURCE_DIR}/level1/l1_retail_test.cu)

# ##################################################################################################
# - L1  tests for quick regression check --------------------------------------------------------------------------
ConfigureTest(ROUTING_L1TEST ${CMAKE_CURRENT_SOURCE_DIR}/level1/l1_routing_test.cu)

# # - ${CMAKE_CURRENT_SOURCE_DIR} unit tests ----------------------------------------------------------------------------
ConfigureTest(ROUTING_UNIT_TEST
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
)
