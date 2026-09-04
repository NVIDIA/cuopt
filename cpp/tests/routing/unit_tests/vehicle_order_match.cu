/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/routing/solve.hpp>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>
#include <map>
#include <vector>

namespace cuopt {
namespace routing {

using i_t = int;
using f_t = float;

/**
 * @brief Test for order vehicle matching with two vehicles and three orders
 */
TEST(vehicle_order_match, two_vehicle_four_orders)
{
  i_t n_vehicles            = 2;
  i_t n_locations           = 4;
  std::vector<f_t> time_mat = {0., 1., 5., 2., 2., 0., 7., 4., 1., 5., 0., 9., 5., 6., 2., 0.};

  std::unordered_map<i_t, std::vector<i_t>> vehicle_order_match{{1, std::vector{0, 2}}};

  raft::handle_t handle;
  cuopt::routing::data_model_view_t<i_t, f_t> data_model(&handle, n_locations, n_vehicles);

  auto time_mat_d = cuopt::device_copy(time_mat, handle.get_stream());
  data_model.add_cost_matrix(time_mat_d.data());

  std::unordered_map<i_t, rmm::device_uvector<i_t>> vehicle_order_match_d;
  for (const auto& [id, orders] : vehicle_order_match) {
    vehicle_order_match_d.emplace(id, cuopt::device_copy(orders, handle.get_stream()));
  }

  for (const auto& [id, orders] : vehicle_order_match_d) {
    data_model.add_vehicle_order_match(id, orders.data(), orders.size());
  }

  auto routing_solution = cuopt::routing::solve(data_model);

  EXPECT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto stream   = handle.get_stream();
  auto route_id = cuopt::host_copy(routing_solution.get_route(), stream);
  auto truck_id = cuopt::host_copy(routing_solution.get_truck_id(), stream);
  for (size_t i = 0; i < route_id.size(); ++i) {
    if (route_id[i] == 3 || route_id[i] == 1) { EXPECT_EQ(truck_id[i], 0); }
  }
}

/**
 * @brief Test for order vehicle matching such that only specific vehicle is allowed to
 * serve each order
 */
TEST(vehicle_order_match, one_order_per_vehicle)
{
  i_t n_vehicles            = 3;
  i_t n_locations           = 4;
  std::vector<f_t> time_mat = {0., 1., 5., 2., 2., 0., 7., 4., 1., 5., 0., 9., 5., 6., 2., 0.};

  std::unordered_map<i_t, std::vector<i_t>> vehicle_order_match{
    {0, std::vector{1}}, {1, std::vector{2}}, {2, std::vector{3}}};

  raft::handle_t handle;
  cuopt::routing::data_model_view_t<i_t, f_t> data_model(&handle, n_locations, n_vehicles);

  auto stream     = handle.get_stream();
  auto time_mat_d = cuopt::device_copy(time_mat, stream);
  data_model.add_cost_matrix(time_mat_d.data());

  std::unordered_map<i_t, rmm::device_uvector<i_t>> vehicle_order_match_d;
  for (const auto& [id, orders] : vehicle_order_match) {
    vehicle_order_match_d.emplace(id, cuopt::device_copy(orders, stream));
  }

  for (const auto& [id, orders] : vehicle_order_match_d) {
    data_model.add_vehicle_order_match(id, orders.data(), orders.size());
  }

  auto routing_solution = cuopt::routing::solve(data_model);

  EXPECT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto route_id = cuopt::host_copy(routing_solution.get_route(), stream);
  auto truck_id = cuopt::host_copy(routing_solution.get_truck_id(), stream);
  for (size_t i = 0; i < route_id.size(); ++i) {
    auto order   = route_id[i];
    auto vehicle = truck_id[i];
    if (order > 0) { EXPECT_EQ(order, vehicle + 1); }
  }
}

/**
 * @brief Test vehicle_order_cost: higher-cost assignments should be avoided.
 *
 * 2 vehicles, 3 orders (depot=0, orders=1,2,3).
 * Vehicle 0 has cost 0 for order 1, 1000 for orders 2 and 3.
 * Vehicle 1 has cost 0 for order 2, 1000 for orders 1 and 3.
 * Vehicle 2 has cost 0 for order 3, 1000 for orders 1 and 2.
 * Optimal: vehicle 0 takes order 1, vehicle 1 takes order 2, vehicle 2 takes order 3.
 */
TEST(vehicle_order_match, vehicle_order_cost_steers_assignment)
{
  i_t n_vehicles            = 3;
  i_t n_locations           = 4;  // depot=0, orders=1,2,3
  std::vector<f_t> time_mat = {0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0.};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  cuopt::routing::data_model_view_t<i_t, f_t> data_model(&handle, n_locations, n_vehicles);

  auto time_mat_d = cuopt::device_copy(time_mat, stream);
  data_model.add_cost_matrix(time_mat_d.data());

  // n_orders = n_locations (depot included means n_orders = n_locations)
  // n_orders as reported to the solver = n_locations (depot at index 0)
  // costs are indexed [0..n_locations-1]: depot gets 0, orders get costs
  const i_t n_orders           = n_locations;
  std::vector<double> costs_v0 = {0., 0., 1000., 1000.};  // prefers order 1
  std::vector<double> costs_v1 = {0., 1000., 0., 1000.};  // prefers order 2
  std::vector<double> costs_v2 = {0., 1000., 1000., 0.};  // prefers order 3

  auto costs_v0_d = cuopt::device_copy(costs_v0, stream);
  auto costs_v1_d = cuopt::device_copy(costs_v1, stream);
  auto costs_v2_d = cuopt::device_copy(costs_v2, stream);

  data_model.set_vehicle_order_cost(0, costs_v0_d.data(), n_orders);
  data_model.set_vehicle_order_cost(1, costs_v1_d.data(), n_orders);
  data_model.set_vehicle_order_cost(2, costs_v2_d.data(), n_orders);

  // Set minimum number of vehicles to 3 to ensure all vehicles are used
  data_model.set_min_vehicles(n_vehicles);

  auto routing_solution = cuopt::routing::solve(data_model);

  EXPECT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto route_id = cuopt::host_copy(routing_solution.get_route(), stream);
  auto truck_id = cuopt::host_copy(routing_solution.get_truck_id(), stream);

  // Build assignment map: order -> vehicle
  std::unordered_map<i_t, i_t> assignment;
  for (size_t i = 0; i < route_id.size(); ++i) {
    if (route_id[i] > 0) { assignment[route_id[i]] = truck_id[i]; }
  }

  // Each order should be served by the vehicle with zero cost for it
  EXPECT_EQ(assignment[1], 0);
  EXPECT_EQ(assignment[2], 1);
  EXPECT_EQ(assignment[3], 2);
}

}  // namespace routing
}  // namespace cuopt
