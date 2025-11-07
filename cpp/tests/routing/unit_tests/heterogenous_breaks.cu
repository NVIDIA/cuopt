/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/utilities/check_constraints.hpp>
#include <routing/utilities/test_utilities.hpp>

#include <cuopt/routing/solve.hpp>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

TEST(heterogenous_breaks, simple_non_uniform)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  int n_orders = 6;
  int n_locs   = n_orders + 1 + 3;
  std::mt19937 rng(42);
  std::uniform_real_distribution<> uni_dis(2.0, 10.0);
  std::vector<float> cost_matrix(n_locs * n_locs, 0.f);
  std::vector<float> time_matrix(n_locs * n_locs, 0.f);
  for (int i = 0; i < n_locs; ++i) {
    for (int j = i + 1; j < n_locs; ++j) {
      cost_matrix[i + j * n_locs] = cost_matrix[j + i * n_locs] = uni_dis(rng);
      time_matrix[i + j * n_locs] = time_matrix[j + i * n_locs] = 10.;
    }
  }

  std::vector<int> order_locs = {1, 2, 3, 4, 5, 6};
  std::vector<int> demand     = {1, 1, 1, 1, 1, 1};
  std::vector<int> cap        = {3, 3};

  std::vector<int> break_locs_v1    = {7};
  std::vector<int> break_locs_v2_b1 = {8};
  std::vector<int> break_locs_v2_b2 = {9};

  int break_earliest_v1 = {15};
  int break_latest_v1   = {25};
  int break_duration_v1 = {5};

  int break_earliest_v2_b1 = {15};
  int break_latest_v2_b1   = {20};
  int break_duration_v2_b1 = {5};

  int break_earliest_v2_b2 = {40};
  int break_latest_v2_b2   = {50};
  int break_duration_v2_b2 = {5};

  std::vector<objective_t> types = {objective_t::COST, objective_t::TRAVEL_TIME};
  std::vector<float> weights     = {1.f, 10.f};

  auto v_cost_matrix = cuopt::device_copy(cost_matrix, stream);
  auto v_time_matrix = cuopt::device_copy(time_matrix, stream);
  auto v_order_locs  = cuopt::device_copy(order_locs, stream);
  auto v_demand      = cuopt::device_copy(demand, stream);
  auto v_cap         = cuopt::device_copy(cap, stream);

  auto v_break_locs_v1    = cuopt::device_copy(break_locs_v1, stream);
  auto v_break_locs_v2_b1 = cuopt::device_copy(break_locs_v2_b1, stream);
  auto v_break_locs_v2_b2 = cuopt::device_copy(break_locs_v2_b2, stream);

  auto v_obj_types   = cuopt::device_copy(types, stream);
  auto v_obj_weights = cuopt::device_copy(weights, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, n_locs, 2, n_orders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_time_matrix.data());
  data_model.set_order_locations(v_order_locs.data());
  data_model.add_capacity_dimension("demand", v_demand.data(), v_cap.data());

  data_model.add_vehicle_break(0,
                               break_earliest_v1,
                               break_latest_v1,
                               break_duration_v1,
                               v_break_locs_v1.data(),
                               v_break_locs_v1.size());

  data_model.add_vehicle_break(1,
                               break_earliest_v2_b1,
                               break_latest_v2_b1,
                               break_duration_v2_b1,
                               v_break_locs_v2_b1.data(),
                               v_break_locs_v2_b1.size());

  data_model.add_vehicle_break(1,
                               break_earliest_v2_b2,
                               break_latest_v2_b2,
                               break_duration_v2_b2,
                               v_break_locs_v2_b2.data(),
                               v_break_locs_v2_b2.size());

  data_model.set_objective_function(v_obj_types.data(), v_obj_weights.data(), v_obj_types.size());

  auto routing_solution = cuopt::routing::solve(data_model);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  // ASSERT_LT(abs(routing_solution.get_total_objective() - 4.0f), 0.001);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
