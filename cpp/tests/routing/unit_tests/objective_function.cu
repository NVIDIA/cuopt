/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/solve.hpp>
#include <routing/node/pdp_node.cuh>
#include <routing/utilities/check_constraints.hpp>
#include <utilities/copy_helpers.hpp>

#include <tuple>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

TEST(objective_function, total_time)
{
  std::vector<float> cost_matrix           = {0, 2, 1, 2, 1, 0, 2, 2, 1, 2, 0, 1, 1, 1, 2, 0};
  std::vector<float> transit_time_matrix   = {0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0};
  std::vector<int> order_locations         = {1, 2, 3};
  std::vector<int> order_earliest          = {2, 5, 12};
  std::vector<int> order_latest            = {1000, 5, 1000};
  std::vector<int> vehicle_start_locations = {0};
  std::vector<int> vehicle_end_locations   = {0};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix             = cuopt::device_copy(cost_matrix, stream);
  auto v_transit_time_matrix     = cuopt::device_copy(transit_time_matrix, stream);
  auto v_order_locations         = cuopt::device_copy(order_locations, stream);
  auto v_order_earliest          = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest            = cuopt::device_copy(order_latest, stream);
  auto v_vehicle_start_locations = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_vehicle_end_locations   = cuopt::device_copy(vehicle_end_locations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 4, 1, 3);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_vehicle_locations(v_vehicle_start_locations.data(),
                                   v_vehicle_end_locations.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(2);

  // run with default objective, cost is setup so that orders 1, 2, 0 are serviced
  // in that order for optimality
  {
    auto routing_solution = cuopt::routing::solve(data_model, settings);

    handle.sync_stream();
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

    auto host_route = cuopt::routing::host_assignment_t(routing_solution);
    // host_route.print();

    auto& nodes = host_route.route;
    ASSERT_EQ(nodes[1], 1);
    ASSERT_EQ(nodes[2], 2);
    ASSERT_EQ(nodes[3], 0);
  }

  // times are set up so that doing the default route will cause a lot of wait time
  // and is not optimal
  {
    std::vector<cuopt::routing::objective_t> objectives = {
      cuopt::routing::objective_t::COST, cuopt::routing::objective_t::TRAVEL_TIME};
    std::vector<float> objective_weights = {0.f, 1.f};

    auto v_objectives        = cuopt::device_copy(objectives, stream);
    auto v_objective_weights = cuopt::device_copy(objective_weights, stream);

    data_model.set_objective_function(
      v_objectives.data(), v_objective_weights.data(), v_objective_weights.size());
    auto routing_solution = cuopt::routing::solve(data_model, settings);

    handle.sync_stream();
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

    auto host_route = cuopt::routing::host_assignment_t(routing_solution);

    // host_route.print();

    auto& nodes = host_route.route;
    ASSERT_EQ(nodes[1], 1);
    ASSERT_EQ(nodes[2], 0);
    ASSERT_EQ(nodes[3], 2);
  }
}

/*
 * arrival times should be shifted to lower the wait times
 */
TEST(objective_function, arrival_times)
{
  std::vector<float> cost_matrix           = {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0};
  std::vector<float> transit_time_matrix   = {0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0};
  std::vector<int> order_locations         = {1, 2, 3};
  std::vector<int> order_earliest          = {10, 20, 30};
  std::vector<int> order_latest            = {1000, 1000, 1000};
  std::vector<int> vehicle_start_locations = {0};
  std::vector<int> vehicle_end_locations   = {0};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix             = cuopt::device_copy(cost_matrix, stream);
  auto v_transit_time_matrix     = cuopt::device_copy(transit_time_matrix, stream);
  auto v_order_locations         = cuopt::device_copy(order_locations, stream);
  auto v_order_earliest          = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest            = cuopt::device_copy(order_latest, stream);
  auto v_vehicle_start_locations = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_vehicle_end_locations   = cuopt::device_copy(vehicle_end_locations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 4, 1, 3);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_vehicle_locations(v_vehicle_start_locations.data(),
                                   v_vehicle_end_locations.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(2);

  std::vector<cuopt::routing::objective_t> objectives = {cuopt::routing::objective_t::COST,
                                                         cuopt::routing::objective_t::TRAVEL_TIME};
  std::vector<float> objective_weights                = {0.f, 1.f};

  auto v_objectives        = cuopt::device_copy(objectives, stream);
  auto v_objective_weights = cuopt::device_copy(objective_weights, stream);

  data_model.set_objective_function(
    v_objectives.data(), v_objective_weights.data(), v_objective_weights.size());
  auto routing_solution = cuopt::routing::solve(data_model, settings);

  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);

  auto& stamp = host_route.stamp;

  // host_route.print();
  // There should not be any wait time
  ASSERT_LE(fabs(stamp[1] - stamp[0]), 5.);
  ASSERT_LE(fabs(stamp[2] - stamp[1]), 5.);
  ASSERT_LE(fabs(stamp[3] - stamp[2]), 5.);
  ASSERT_LE(fabs(stamp[4] - stamp[3]), 5.);
}
}  // namespace test
}  // namespace routing
}  // namespace cuopt
