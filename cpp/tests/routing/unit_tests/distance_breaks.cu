/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/node/distance_node.cuh>
#include <routing/utilities/check_constraints.hpp>
#include <routing/utilities/test_utilities.hpp>

#include <cuopt/routing/solve.hpp>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>
#include <limits>
#include <unordered_map>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

namespace {

using distance_node = detail::distance_node_t<int, float>;

struct test_route {
  std::vector<distance_node> nodes;
  std::vector<float> arcs;
  detail::VehicleInfo<float> vehicle_info{};

  void run_passes()
  {
    auto n_arcs = static_cast<int>(arcs.size());
    for (int i = 0; i < n_arcs; ++i) {
      nodes[i].calculate_forward(nodes[i + 1], arcs[i]);
    }
    for (int i = n_arcs; i > 0; --i) {
      nodes[i].calculate_backward(nodes[i - 1], arcs[i - 1]);
    }
  }
};

// windows.size() must equal arcs.size() + 1; window {0, 1e18} means unconstrained.
test_route make_route(std::vector<float> arcs,
                      std::vector<std::pair<double, double>> windows,
                      float max_cost = std::numeric_limits<float>::max())
{
  test_route r;
  r.arcs                  = std::move(arcs);
  auto n_nodes            = r.arcs.size() + 1;
  r.vehicle_info.max_cost = max_cost;
  r.nodes.resize(n_nodes);
  for (size_t i = 0; i < n_nodes; ++i) {
    r.nodes[i].window_start = windows[i].first;
    r.nodes[i].window_end   = windows[i].second;
  }
  r.nodes.back().distance_window_backward = max_cost;
  return r;
}

}  // namespace

TEST(distance_node, forward_propagation)
{
  // depot -> break(window [0, 60]) -> A -> end. Arcs 80, 600, 400. Overshoots break by 20.
  auto r = make_route({80.f, 600.f, 400.f}, {{0., 1e18}, {0., 60.}, {0., 1e18}, {0., 1e18}});
  r.run_passes();

  EXPECT_DOUBLE_EQ(r.nodes[0].distance_forward, 0.);
  EXPECT_DOUBLE_EQ(r.nodes[1].distance_forward, 80.);
  EXPECT_DOUBLE_EQ(r.nodes[2].distance_forward, 680.);
  EXPECT_DOUBLE_EQ(r.nodes[3].distance_forward, 1080.);

  EXPECT_DOUBLE_EQ(r.nodes[0].distance_window_forward, 0.);
  EXPECT_DOUBLE_EQ(r.nodes[1].distance_window_forward, 60.);
  EXPECT_DOUBLE_EQ(r.nodes[2].distance_window_forward, 660.);
  EXPECT_DOUBLE_EQ(r.nodes[3].distance_window_forward, 1060.);

  EXPECT_DOUBLE_EQ(r.nodes[0].excess_forward, 0.);
  EXPECT_DOUBLE_EQ(r.nodes[1].excess_forward, 20.);
  EXPECT_DOUBLE_EQ(r.nodes[2].excess_forward, 20.);
  EXPECT_DOUBLE_EQ(r.nodes[3].excess_forward, 20.);
}

TEST(distance_node, backward_propagation)
{
  // Same route as forward_propagation, max_cost = 800 (route total 1080 overshoots by 280).
  auto r = make_route({80.f, 600.f, 400.f},
                      {{0., 1e18}, {0., 60.}, {0., 1e18}, {0., 1e18}},
                      /*max_cost=*/800.f);
  r.run_passes();

  EXPECT_DOUBLE_EQ(r.nodes[3].distance_backward, 0.);
  EXPECT_DOUBLE_EQ(r.nodes[2].distance_backward, 400.);
  EXPECT_DOUBLE_EQ(r.nodes[1].distance_backward, 1000.);
  EXPECT_DOUBLE_EQ(r.nodes[0].distance_backward, 1080.);

  EXPECT_DOUBLE_EQ(r.nodes[3].distance_window_backward, 800.);
  EXPECT_DOUBLE_EQ(r.nodes[2].distance_window_backward, 400.);
  EXPECT_DOUBLE_EQ(r.nodes[1].distance_window_backward, 0.);
  EXPECT_DOUBLE_EQ(r.nodes[1].excess_backward, 200.);
  EXPECT_DOUBLE_EQ(r.nodes[0].distance_window_backward, 0.);
  EXPECT_DOUBLE_EQ(r.nodes[0].excess_backward, 280.);
}

// combine(node[k], node[k+1]) must return the same value at every split point of a fixed route.
TEST(distance_node, combine_invariant_feasible)
{
  auto r = make_route({50.f, 55.f, 100.f},
                      {{0., 1e18}, {0., 60.}, {0., 1e18}, {0., 1e18}},
                      /*max_cost=*/1000.f);
  r.run_passes();

  for (size_t k = 0; k + 1 < r.nodes.size(); ++k) {
    double c = distance_node::combine(r.nodes[k], r.nodes[k + 1], r.vehicle_info, r.arcs[k]);
    EXPECT_DOUBLE_EQ(c, 0.) << "split (" << k << ", " << (k + 1) << ") got " << c;
  }
}

TEST(distance_node, combine_invariant_window_violation)
{
  auto r = make_route({80.f, 600.f, 400.f},
                      {{0., 1e18}, {0., 60.}, {0., 1e18}, {0., 1e18}},
                      /*max_cost=*/800.f);
  r.run_passes();

  double reference = distance_node::combine(r.nodes[0], r.nodes[1], r.vehicle_info, r.arcs[0]);
  EXPECT_GT(reference, 0.);
  for (size_t k = 1; k + 1 < r.nodes.size(); ++k) {
    double c = distance_node::combine(r.nodes[k], r.nodes[k + 1], r.vehicle_info, r.arcs[k]);
    EXPECT_DOUBLE_EQ(c, reference) << "split (" << k << ", " << (k + 1) << ") = " << c
                                   << " differs from reference " << reference;
  }
}

TEST(distance_node, combine_invariant_max_cost_only)
{
  auto r = make_route({400.f, 300.f, 400.f},
                      {{0., 1e18}, {0., 1e18}, {0., 1e18}, {0., 1e18}},
                      /*max_cost=*/1000.f);
  r.run_passes();

  double reference = distance_node::combine(r.nodes[0], r.nodes[1], r.vehicle_info, r.arcs[0]);
  EXPECT_DOUBLE_EQ(reference, 100.);  // total 1100, max_cost 1000.
  for (size_t k = 1; k + 1 < r.nodes.size(); ++k) {
    double c = distance_node::combine(r.nodes[k], r.nodes[k + 1], r.vehicle_info, r.arcs[k]);
    EXPECT_DOUBLE_EQ(c, reference);
  }
}

TEST(distance_node, compute_cost_combine_consistency)
{
  auto r = make_route({80.f, 600.f, 400.f},
                      {{0., 1e18}, {0., 60.}, {0., 1e18}, {0., 1e18}},
                      /*max_cost=*/800.f);
  r.run_passes();

  auto const& end_node = r.nodes.back();
  double boundary =
    std::max(0., end_node.distance_window_forward - end_node.distance_window_backward);
  double total = end_node.excess_forward + boundary;

  double combine_at_first =
    distance_node::combine(r.nodes[0], r.nodes[1], r.vehicle_info, r.arcs[0]);

  EXPECT_DOUBLE_EQ(total, combine_at_first);
}

TEST(distance_node, get_cost_combine_consistency)
{
  auto r = make_route({80.f, 600.f, 400.f},
                      {{0., 1e18}, {0., 60.}, {0., 1e18}, {0., 1e18}},
                      /*max_cost=*/800.f);
  r.run_passes();

  detail::cost_dimension_info_t dim_info;
  dim_info.has_max_constraint  = true;
  dim_info.has_distance_window = true;

  for (size_t k = 0; k + 1 < r.nodes.size(); ++k) {
    auto next_copy = r.nodes[k + 1];
    r.nodes[k].calculate_forward(next_copy, r.arcs[k]);

    detail::objective_cost_t obj_cost;
    detail::infeasible_cost_t inf_cost;
    next_copy.get_cost(r.nodes[k], r.vehicle_info, dim_info, obj_cost, inf_cost);
    double get_cost_total = inf_cost[detail::dim_t::DIST];

    double combine_value =
      distance_node::combine(r.nodes[k], r.nodes[k + 1], r.vehicle_info, r.arcs[k]);

    EXPECT_DOUBLE_EQ(get_cost_total, combine_value)
      << "split (" << k << ", " << (k + 1) << "): get_cost = " << get_cost_total
      << ", combine = " << combine_value;
  }
}

// depot=0, orders=1-2, optional charging stations=3-4 (used in 5x5 tests)
// clang-format off
static std::vector<float> cost_matrix_3x3 = {
  0, 1, 1,
  1, 0, 1,
  1, 1, 0,
};
static std::vector<float> cost_matrix_5x5 = {
  0, 1, 1, 1, 1,
  1, 0, 1, 1, 1,
  1, 1, 0, 1, 1,
  1, 1, 1, 0, 1,
  1, 1, 1, 1, 0,
};
// clang-format on

TEST(distance_breaks, default_case)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix = cuopt::device_copy(cost_matrix_3x3, stream);
  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_distance_break(0, 0.f, 2.f, 1, nullptr, 0);
  data_model.add_distance_break(1, 0.f, 2.f, 1, nullptr, 0);
  data_model.set_min_vehicles(2);

  auto routing_solution = cuopt::routing::solve(data_model);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);
}

TEST(distance_breaks, with_charging_stations)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::vector<int> order_locations   = {1, 2};
  std::vector<int> charging_stations = {3, 4};

  auto v_cost_matrix       = cuopt::device_copy(cost_matrix_5x5, stream);
  auto v_order_locations   = cuopt::device_copy(order_locations, stream);
  auto v_charging_stations = cuopt::device_copy(charging_stations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 5, 2, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.add_distance_break(
    0, 0.f, 2.f, 1, v_charging_stations.data(), (int)v_charging_stations.size());
  data_model.add_distance_break(
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

TEST(distance_breaks, multi_cycle)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  // Two vehicles, each with two charge cycles: [0, 2) and [2, 4).
  std::vector<int> order_locations = {1, 2};

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix_5x5, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 5, 2, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());

  for (int vid = 0; vid < 2; ++vid) {
    data_model.add_distance_break(vid, 0.f, 2.f, 1, nullptr, 0);
    data_model.add_distance_break(vid, 2.f, 4.f, 1, nullptr, 0);
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

// The cheap customer-first route 0→1→2→0 (cost 106) would put the charger at cumulative
// 105 > d_max=60 — infeasible. The solver must choose the more expensive charger-first
// ordering 0→2→1→0 (cost 205) where the charger lands at 50 ≤ 60.
TEST(distance_breaks, break_distance_window_enforced)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  // clang-format off
  std::vector<float> cost_matrix_3 = {
    0,   100, 50,
    100, 0,   5,
    1,   55,  0,
  };
  // clang-format on
  std::vector<int> order_locations   = {1};
  std::vector<int> charging_stations = {2};

  auto v_cost_matrix       = cuopt::device_copy(cost_matrix_3, stream);
  auto v_order_locations   = cuopt::device_copy(order_locations, stream);
  auto v_charging_stations = cuopt::device_copy(charging_stations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 1, 1);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.add_distance_break(0, 0.f, 60.f, 0, v_charging_stations.data(), 1);

  auto settings = cuopt::routing::solver_settings_t<int, float>{};
  settings.set_time_limit(10);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  host_assignment_t<int> h(routing_solution);

  float cumulative = 0.f;
  int prev_loc     = 0;
  bool found_break = false;
  for (size_t i = 0; i < h.locations.size(); ++i) {
    int loc = h.locations[i];
    cumulative += cost_matrix_3[prev_loc * 3 + loc];
    if (static_cast<node_type_t>(h.node_types[i]) == node_type_t::BREAK) {
      found_break = true;
      EXPECT_LE(cumulative, 60.f) << "break at cumulative distance " << cumulative
                                  << " exceeds d_max=60";
    }
    prev_loc = loc;
  }
  EXPECT_TRUE(found_break) << "no break found in solution";
}

TEST(distance_breaks, mixed_fleet)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix = cuopt::device_copy(cost_matrix_3x3, stream);
  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_distance_break(0, 0.f, 2.f, 1, nullptr, 0);
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
