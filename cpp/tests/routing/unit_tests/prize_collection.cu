/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST(prize_collection, simple)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::vector<float> cost_matrix = {0, 1, 1, 1, 0, 1, 1, 1, 0};

  std::vector<int> cap = {1};

  std::vector<int> order_locations = {1, 2};
  std::vector<float> order_prizes  = {1.f, 10.f};
  std::vector<int> demand          = {1, 1};
  std::vector<int> order_earliest  = {0, 0};
  std::vector<int> order_latest    = {1000, 1000};

  auto v_cost_matrix = cuopt::device_copy(cost_matrix, stream);

  auto v_cap = cuopt::device_copy(cap, stream);

  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_order_prizes    = cuopt::device_copy(order_prizes, stream);

  auto v_demand         = cuopt::device_copy(demand, stream);
  auto v_order_earliest = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest   = cuopt::device_copy(order_latest, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 1, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());

  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_prizes(v_order_prizes.data());
  data_model.add_capacity_dimension("dim", v_demand.data(), v_cap.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(1);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  host_assignment_t<int> h_routing_solution(routing_solution);

  check_route(data_model, h_routing_solution);

  // h_routing_solution.print();

  // only order 1 should be served, order 0 will not be served because of constraints
  ASSERT_EQ(h_routing_solution.route[1], 1);
}

TEST(prize_collection, simple_pdp)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::vector<float> cost_matrix = {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                                    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0};

  std::vector<int> order_locations  = {1, 2, 3, 4};
  std::vector<int> pickup_indices   = {0, 2};
  std::vector<int> delivery_indices = {1, 3};

  std::vector<float> order_prizes = {1.f, 1.f, 10.f, 10.f};

  std::vector<int> order_earliest = {0, 0, 0, 0};
  std::vector<int> order_latest   = {1000, 1000, 1000, 1000};

  std::vector<float> vehicle_max_costs = {4.f};

  auto v_cost_matrix = cuopt::device_copy(cost_matrix, stream);

  auto v_order_locations  = cuopt::device_copy(order_locations, stream);
  auto v_pickup_indices   = cuopt::device_copy(pickup_indices, stream);
  auto v_delivery_indices = cuopt::device_copy(delivery_indices, stream);

  auto v_order_prizes = cuopt::device_copy(order_prizes, stream);

  auto v_order_earliest = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest   = cuopt::device_copy(order_latest, stream);

  auto v_vehicle_max_costs = cuopt::device_copy(vehicle_max_costs, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 5, 1, 4);
  data_model.add_cost_matrix(v_cost_matrix.data());

  data_model.set_order_locations(v_order_locations.data());
  data_model.set_pickup_delivery_pairs(v_pickup_indices.data(), v_delivery_indices.data());
  data_model.set_order_prizes(v_order_prizes.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());

  data_model.set_vehicle_max_costs(v_vehicle_max_costs.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(1);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  host_assignment_t<int> h_routing_solution(routing_solution);

  check_route(data_model, h_routing_solution);

  // h_routing_solution.print();

  // only order 1 should be served, order 0 will not be served because of constraints
  ASSERT_EQ(h_routing_solution.route[1], 2);
  ASSERT_EQ(h_routing_solution.route[2], 3);
  // 2 nodes (i.e. one pdp request) is not served
  ASSERT_EQ(h_routing_solution.unserviced_nodes.size(), 2u);
}

TEST(prize_collection, two_vs_one)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::vector<float> cost_matrix = {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0};

  std::vector<int> cap = {2};

  std::vector<int> order_locations = {1, 2, 3};
  std::vector<float> order_prizes  = {5.f, 10.f, 5.f};
  std::vector<int> demand          = {1, 2, 1};

  std::vector<int> order_earliest = {0, 0, 0};
  std::vector<int> order_latest   = {1000, 1000, 1000};

  auto v_cost_matrix = cuopt::device_copy(cost_matrix, stream);

  auto v_cap = cuopt::device_copy(cap, stream);

  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_order_prizes    = cuopt::device_copy(order_prizes, stream);

  auto v_demand         = cuopt::device_copy(demand, stream);
  auto v_order_earliest = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest   = cuopt::device_copy(order_latest, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 4, 1, 3);
  data_model.add_cost_matrix(v_cost_matrix.data());

  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_prizes(v_order_prizes.data());
  data_model.add_capacity_dimension("dim", v_demand.data(), v_cap.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(1);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  host_assignment_t<int> h_routing_solution(routing_solution);

  check_route(data_model, h_routing_solution);

  // h_routing_solution.print();

  // only order 1 should be served, serving order 0 & 2 will have  more cost
  ASSERT_EQ(h_routing_solution.route[1], 1);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
