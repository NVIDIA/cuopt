/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  auto route_id = cuopt::host_copy(routing_solution.get_route());
  auto truck_id = cuopt::host_copy(routing_solution.get_truck_id());
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

  auto route_id = cuopt::host_copy(routing_solution.get_route());
  auto truck_id = cuopt::host_copy(routing_solution.get_truck_id());
  for (size_t i = 0; i < route_id.size(); ++i) {
    auto order   = route_id[i];
    auto vehicle = truck_id[i];
    if (order > 0) { EXPECT_EQ(order, vehicle + 1); }
  }
}

}  // namespace routing
}  // namespace cuopt
