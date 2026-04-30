/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/utilities/check_constraints.hpp>
#include <routing/utilities/test_utilities.hpp>

#include <cuopt/routing/solve.hpp>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

// 3-location problem shared by the basic tests
static std::vector<float> cost_matrix_3x3 = {0, 1, 1, 1, 0, 1, 1, 1, 0};

// 5-location problem used by the charging-stations and multi-cycle tests:
// depot=0, orders=1-2, dedicated charging stations=3-4
static std::vector<float> cost_matrix_5x5 = {
  0, 1, 1, 1, 1,
  1, 0, 1, 1, 1,
  1, 1, 0, 1, 1,
  1, 1, 1, 0, 1,
  1, 1, 1, 1, 0
};

// validates that vehicles belong to their breaks
TEST(ev_breaks, default_case)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix = cuopt::device_copy(cost_matrix_3x3, stream);
  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_vehicle_ev_break(0, 0.f, 2.f, 1, nullptr, 0);
  data_model.add_vehicle_ev_break(1, 0.f, 2.f, 1, nullptr, 0);
  data_model.set_min_vehicles(2);

  auto routing_solution = cuopt::routing::solve(data_model);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);
}

// charging stations are used as EV breaks if specified
TEST(ev_breaks, with_charging_stations)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  // Orders are at locs 1-2; charging stations are at locs 3-4 only
  std::vector<int> order_locations   = {1, 2};
  std::vector<int> charging_stations = {3, 4};

  auto v_cost_matrix      = cuopt::device_copy(cost_matrix_5x5, stream);
  auto v_order_locations  = cuopt::device_copy(order_locations, stream);
  auto v_charging_stations = cuopt::device_copy(charging_stations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 5, 2, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.add_vehicle_ev_break(
    0, 0.f, 2.f, 1, v_charging_stations.data(), (int)v_charging_stations.size());
  data_model.add_vehicle_ev_break(
    1, 0.f, 2.f, 1, v_charging_stations.data(), (int)v_charging_stations.size());
  data_model.set_min_vehicles(2);

  auto settings = cuopt::routing::solver_settings_t<int, float>{};
  settings.set_time_limit(10);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);

  for (size_t i = 0; i < h_routing_solution.node_types.size(); ++i) {
    if ((node_type_t)h_routing_solution.node_types[i] == node_type_t::BREAK) {
      auto loc = h_routing_solution.locations[i];
      ASSERT_TRUE(loc == 3 || loc == 4);
    }
  }
}

// multiple charging cycles are handled correctly
TEST(ev_breaks, multi_cycle)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  // Two vehicles each requiring two charge cycles:
  // cycle 0 covers [0, 2), cycle 1 covers [2, 4)
  std::vector<int> order_locations = {1, 2};

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix_5x5, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 5, 2, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());

  for (int vid = 0; vid < 2; ++vid) {
    data_model.add_vehicle_ev_break(vid, 0.f, 2.f, 1, nullptr, 0);
    data_model.add_vehicle_ev_break(vid, 2.f, 4.f, 1, nullptr, 0);
  }
  data_model.set_min_vehicles(2);

  auto settings = cuopt::routing::solver_settings_t<int, float>{};
  settings.set_time_limit(10);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);

  // Every vehicle that appears in the solution must carry exactly 2 breaks
  std::unordered_map<int, int> break_count;
  for (size_t i = 0; i < h_routing_solution.node_types.size(); ++i) {
    if ((node_type_t)h_routing_solution.node_types[i] == node_type_t::BREAK) {
      break_count[h_routing_solution.truck_id[i]]++;
    }
  }
  for (auto const& [vid, cnt] : break_count) {
    ASSERT_EQ(cnt, 2);
  }
}

// only vehicle with EV break specified has such break assigned
TEST(ev_breaks, mixed_fleet)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  // Only vehicle 0 has an EV break; vehicle 1 does not
  auto v_cost_matrix = cuopt::device_copy(cost_matrix_3x3, stream);
  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_vehicle_ev_break(0, 0.f, 2.f, 1, nullptr, 0);
  data_model.set_min_vehicles(2);

  auto settings = cuopt::routing::solver_settings_t<int, float>{};
  settings.set_time_limit(10);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);

  for (size_t i = 0; i < h_routing_solution.node_types.size(); ++i) {
    if ((node_type_t)h_routing_solution.node_types[i] == node_type_t::BREAK) {
      ASSERT_EQ(h_routing_solution.truck_id[i], 0);
    }
  }
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
