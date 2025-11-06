/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/utilities/check_constraints.hpp>
#include <routing/utilities/test_utilities.hpp>

#include <cuopt/routing/solve.hpp>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

TEST(horizontal_loading, route_sizes)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  int nlocations = 7;
  int norders    = 6;
  int nvehicles  = 2;

  std::vector<float> cost_matrix_cheap(nlocations * nlocations, 1),
    cost_matrix_expensive(nlocations * nlocations, 10);

  for (int i = 0; i < nlocations; ++i) {
    cost_matrix_cheap[i * nlocations + i]     = 0;
    cost_matrix_expensive[i * nlocations + i] = 0;
  }

  std::vector<uint8_t> vehicle_types = {0, 1};
  std::vector<int> vehicle_capacity  = {100, 100};

  std::vector<int> order_locations                  = {1, 2, 3, 4, 5, 6};
  std::vector<int> order_demand                     = {1, 1, 1, 1, 1, 1};
  std::vector<int> order_vehicle_match_first_order  = {0};
  std::vector<int> order_vehicle_match_second_order = {1};

  std::vector<cuopt::routing::objective_t> objectives = {
    cuopt::routing::objective_t::VARIANCE_ROUTE_SIZE};
  std::vector<float> objective_weights = {1000.};

  auto v_cost_matrix_cheap     = cuopt::device_copy(cost_matrix_cheap, stream);
  auto v_cost_matrix_expensive = cuopt::device_copy(cost_matrix_expensive, stream);

  auto v_vehicle_types    = cuopt::device_copy(vehicle_types, stream);
  auto v_vehicle_capacity = cuopt::device_copy(vehicle_capacity, stream);

  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_order_demand    = cuopt::device_copy(order_demand, stream);
  auto v_order_vehicle_match_first_order =
    cuopt::device_copy(order_vehicle_match_first_order, stream);
  auto v_order_vehicle_match_second_order =
    cuopt::device_copy(order_vehicle_match_second_order, stream);

  auto v_objectives        = cuopt::device_copy(objectives, stream);
  auto v_objective_weights = cuopt::device_copy(objective_weights, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);

  data_model.add_cost_matrix(v_cost_matrix_cheap.data(), 0);
  data_model.add_cost_matrix(v_cost_matrix_expensive.data(), 1);
  data_model.set_vehicle_types(v_vehicle_types.data());

  data_model.set_order_locations(v_order_locations.data());
  data_model.add_order_vehicle_match(
    0, v_order_vehicle_match_first_order.data(), v_order_vehicle_match_first_order.size());
  data_model.add_order_vehicle_match(
    1, v_order_vehicle_match_second_order.data(), v_order_vehicle_match_second_order.size());

  data_model.add_capacity_dimension("demand", v_order_demand.data(), v_vehicle_capacity.data());

  data_model.set_objective_function(
    v_objectives.data(), v_objective_weights.data(), v_objective_weights.size());

  auto routing_solution = cuopt::routing::solve(data_model);

  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  ASSERT_EQ(routing_solution.get_vehicle_count(), 2);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);

  std::vector<int> route_sizes(nvehicles, 0);
  for (auto& vehicle_id : h_routing_solution.truck_id) {
    route_sizes[vehicle_id]++;
  }

  for (auto& route_size : route_sizes) {
    ASSERT_EQ(route_size, 5);
  }
}

TEST(horizontal_loading, route_service_times)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  int nlocations = 7;
  int norders    = 6;
  int nvehicles  = 2;

  std::vector<float> cost_matrix_cheap(nlocations * nlocations, 1),
    cost_matrix_expensive(nlocations * nlocations, 10);

  for (int i = 0; i < nlocations; ++i) {
    cost_matrix_cheap[i * nlocations + i]     = 0;
    cost_matrix_expensive[i * nlocations + i] = 0;
  }

  std::vector<uint8_t> vehicle_types = {0, 1};
  std::vector<int> vehicle_capacity  = {100, 100};

  std::vector<int> order_locations = {1, 2, 3, 4, 5, 6};
  std::vector<int> order_demand    = {1, 1, 1, 1, 1, 1};
  std::vector<int> order_earliest  = {0, 0, 0, 0, 0, 0};
  std::vector<int> order_latest    = {1000, 1000, 1000, 1000, 1000, 1000};
  std::vector<int> order_service   = {50, 10, 10, 10, 10, 10};

  std::vector<int> order_vehicle_match_first_order  = {0};
  std::vector<int> order_vehicle_match_second_order = {1};

  std::vector<cuopt::routing::objective_t> objectives = {
    cuopt::routing::objective_t::VARIANCE_ROUTE_SERVICE_TIME};
  std::vector<float> objective_weights = {1000.};

  auto v_cost_matrix_cheap     = cuopt::device_copy(cost_matrix_cheap, stream);
  auto v_cost_matrix_expensive = cuopt::device_copy(cost_matrix_expensive, stream);

  auto v_vehicle_types    = cuopt::device_copy(vehicle_types, stream);
  auto v_vehicle_capacity = cuopt::device_copy(vehicle_capacity, stream);

  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_order_demand    = cuopt::device_copy(order_demand, stream);
  auto v_order_earliest  = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest    = cuopt::device_copy(order_latest, stream);
  auto v_order_service   = cuopt::device_copy(order_service, stream);

  auto v_order_vehicle_match_first_order =
    cuopt::device_copy(order_vehicle_match_first_order, stream);
  auto v_order_vehicle_match_second_order =
    cuopt::device_copy(order_vehicle_match_second_order, stream);

  auto v_objectives        = cuopt::device_copy(objectives, stream);
  auto v_objective_weights = cuopt::device_copy(objective_weights, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);

  data_model.add_cost_matrix(v_cost_matrix_cheap.data(), 0);
  data_model.add_cost_matrix(v_cost_matrix_expensive.data(), 1);
  data_model.set_vehicle_types(v_vehicle_types.data());

  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_order_service_times(v_order_service.data());

  data_model.add_order_vehicle_match(
    0, v_order_vehicle_match_first_order.data(), v_order_vehicle_match_first_order.size());
  data_model.add_order_vehicle_match(
    1, v_order_vehicle_match_second_order.data(), v_order_vehicle_match_second_order.size());

  data_model.add_capacity_dimension("demand", v_order_demand.data(), v_vehicle_capacity.data());

  data_model.set_objective_function(
    v_objectives.data(), v_objective_weights.data(), v_objective_weights.size());

  cuopt::routing::solver_settings_t<int, float> settings;
  auto routing_solution = cuopt::routing::solve(data_model, settings);

  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  ASSERT_EQ(routing_solution.get_vehicle_count(), 2);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);

  std::vector<int> route_sizes(nvehicles, 0);
  for (auto& vehicle_id : h_routing_solution.truck_id) {
    route_sizes[vehicle_id]++;
  }

  ASSERT_EQ(route_sizes[0], 3);
  ASSERT_EQ(route_sizes[1], 7);
}
}  // namespace test
}  // namespace routing
}  // namespace cuopt
