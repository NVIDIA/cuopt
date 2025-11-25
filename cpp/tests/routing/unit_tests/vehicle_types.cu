/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/solve.hpp>
#include <routing/utilities/check_constraints.hpp>
#include <utilities/copy_helpers.hpp>

#include <tuple>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

TEST(vehicle_types, non_contiguous_ids)
{
  std::vector<float> cost_matrix     = {0, 1, 1, 1, 0, 1, 1, 1, 0};
  std::vector<float> time_matrix     = {0, 2, 2, 2, 0, 2, 2, 2, 0};
  std::vector<uint8_t> vehicle_types = {18, 18, 4, 3, 18};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix   = cuopt::device_copy(cost_matrix, stream);
  auto v_cost_matrix_1 = cuopt::device_copy(cost_matrix, stream);
  auto v_cost_matrix_2 = cuopt::device_copy(cost_matrix, stream);
  auto v_time_matrix   = cuopt::device_copy(time_matrix, stream);
  auto v_vehicle_types = cuopt::device_copy(vehicle_types, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 4);
  data_model.add_cost_matrix(v_cost_matrix.data(), 18);
  data_model.add_cost_matrix(v_cost_matrix_1.data(), 4);
  data_model.add_cost_matrix(v_cost_matrix_2.data(), 3);
  data_model.add_transit_time_matrix(v_time_matrix.data(), 18);
  data_model.set_vehicle_types(v_vehicle_types.data());

  auto routing_solution = cuopt::routing::solve(data_model);
  handle.sync_stream();
  check_route(data_model, routing_solution);
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
}

TEST(vehicle_types, simple)
{
  std::vector<float> cost_matrix_t1  = {0, 1, 1, 1, 0, 1, 1, 1, 0};
  std::vector<float> time_matrix_t1  = {0, 5, 5, 5, 0, 5, 5, 5, 0};
  std::vector<float> cost_matrix_t2  = {0, 10, 10, 10, 0, 10, 10, 10, 0};
  std::vector<float> time_matrix_t2  = {0, 8, 8, 8, 0, 8, 8, 8, 0};
  std::vector<int> order_locations   = {1, 2, 2};
  std::vector<uint8_t> vehicle_types = {0, 1};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix_t1  = cuopt::device_copy(cost_matrix_t1, stream);
  auto v_time_matrix_t1  = cuopt::device_copy(time_matrix_t1, stream);
  auto v_cost_matrix_t2  = cuopt::device_copy(cost_matrix_t2, stream);
  auto v_time_matrix_t2  = cuopt::device_copy(time_matrix_t2, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_vehicle_types   = cuopt::device_copy(vehicle_types, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 2, 2);
  data_model.add_cost_matrix(v_cost_matrix_t1.data(), 0);
  data_model.add_cost_matrix(v_cost_matrix_t2.data(), 1);
  data_model.add_transit_time_matrix(v_time_matrix_t1.data(), 0);
  data_model.add_transit_time_matrix(v_time_matrix_t2.data(), 1);
  data_model.set_vehicle_types(v_vehicle_types.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_min_vehicles(2);

  auto routing_solution = cuopt::routing::solve(data_model);
  handle.sync_stream();

  check_route(data_model, routing_solution);
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  // FIXME: Determinism PR
  // ASSERT_EQ(routing_solution.get_total_objective(), 32);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
