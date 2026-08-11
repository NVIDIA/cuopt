/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/node/distance_node.cuh>
#include <routing/route/distance_route.cuh>
#include <routing/utilities/check_constraints.hpp>
#include <routing/utilities/test_utilities.hpp>

#include <cuopt/routing/solve.hpp>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <unordered_map>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

namespace {

using distance_node         = detail::distance_node_t<int, float>;
using distance_route        = detail::distance_route_t<int, float>;
constexpr auto DISTANCE_INF = detail::DISTANCE_WINDOW_INFINITY;

template <typename T, size_t N>
auto copy_array_to_device(std::array<T, N> const& values, rmm::cuda_stream_view stream)
{
  rmm::device_uvector<T> result(values.size(), stream);
  raft::copy(result.data(), values.data(), values.size(), stream);
  return result;
}

__global__ void compute_distance_route_cost(distance_route::view_t route, double* result)
{
  detail::objective_cost_t objective_cost;
  detail::infeasible_cost_t infeasible_cost;
  detail::VehicleInfo<float> vehicle_info;
  route.compute_cost(vehicle_info, 0, objective_cost, infeasible_cost);
  result[0] = objective_cost[objective_t::DISTANCE_BREAK_COST];
}

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

// windows.size() must equal arcs.size() + 1;
// {0, DISTANCE_INF} means unconstrained.
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
  r.nodes.back().distance_window_backward = DISTANCE_INF;
  return r;
}

}  // namespace

// Forward sweep clamps cumulative distance at hard upper bounds and accumulates excess.
TEST(distance_node, forward_propagation)
{
  auto r = make_route({80.f, 600.f, 400.f},
                      {{0., DISTANCE_INF}, {0., 60.}, {0., DISTANCE_INF}, {0., DISTANCE_INF}});
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

// Multiple early breaks contribute the route's maximum shortfall, not a sum of hinges.
TEST(distance_node, early_arrival_cost_is_maximum_per_route)
{
  auto r = make_route({10.f, 10.f, 0.f},
                      {{0., DISTANCE_INF}, {50., 100.}, {80., 100.}, {0., DISTANCE_INF}},
                      /*max_cost=*/1000.f);
  r.run_passes();

  EXPECT_DOUBLE_EQ(r.nodes[1].distance_break_cost_forward, 40.);
  EXPECT_DOUBLE_EQ(r.nodes[2].distance_break_cost_forward, 60.);
  EXPECT_DOUBLE_EQ(r.nodes[3].distance_break_cost_forward, 60.);

  detail::cost_dimension_info_t dim_info;
  dim_info.has_distance_window     = true;
  dim_info.has_distance_break_cost = true;

  for (size_t k = 0; k + 1 < r.nodes.size(); ++k) {
    auto next_copy = r.nodes[k + 1];
    r.nodes[k].calculate_forward(next_copy, r.arcs[k]);

    detail::objective_cost_t obj_cost;
    detail::infeasible_cost_t inf_cost;
    next_copy.get_cost(r.nodes[k], r.vehicle_info, dim_info, obj_cost, inf_cost);

    EXPECT_DOUBLE_EQ(obj_cost[objective_t::DISTANCE_BREAK_COST], 60.)
      << "split (" << k << ", " << (k + 1) << ")";
    EXPECT_DOUBLE_EQ(inf_cost[detail::dim_t::DIST], 0.);
  }
}

// A soft lower-bound correction must not shift the independently propagated hard upper state.
TEST(distance_node, early_arrival_does_not_create_later_upper_excess)
{
  auto r = make_route({0.f, 40.f, 0.f},
                      {{0., DISTANCE_INF}, {100., 200.}, {0., 30.}, {0., DISTANCE_INF}},
                      /*max_cost=*/1000.f);
  r.run_passes();

  EXPECT_DOUBLE_EQ(r.nodes[1].distance_forward, 0.);
  EXPECT_DOUBLE_EQ(r.nodes[2].distance_forward, 40.);
  EXPECT_DOUBLE_EQ(r.nodes[2].distance_window_forward, 30.);
  EXPECT_DOUBLE_EQ(r.nodes[2].distance_break_cost_forward, 100.);
  EXPECT_DOUBLE_EQ(r.nodes[2].excess_forward, 10.);

  detail::cost_dimension_info_t dim_info;
  dim_info.has_distance_window     = true;
  dim_info.has_distance_break_cost = true;

  for (size_t k = 0; k + 1 < r.nodes.size(); ++k) {
    auto next_copy = r.nodes[k + 1];
    r.nodes[k].calculate_forward(next_copy, r.arcs[k]);

    detail::objective_cost_t obj_cost;
    detail::infeasible_cost_t inf_cost;
    next_copy.get_cost(r.nodes[k], r.vehicle_info, dim_info, obj_cost, inf_cost);

    EXPECT_DOUBLE_EQ(obj_cost[objective_t::DISTANCE_BREAK_COST], 100.)
      << "split (" << k << ", " << (k + 1) << ")";
    EXPECT_DOUBLE_EQ(inf_cost[detail::dim_t::DIST], 10.)
      << "split (" << k << ", " << (k + 1) << ")";
    EXPECT_DOUBLE_EQ(distance_node::combine(r.nodes[k], r.nodes[k + 1], r.vehicle_info, r.arcs[k]),
                     10.)
      << "split (" << k << ", " << (k + 1) << ")";
  }
}

// Backward sweep propagates the latest cumulative distance allowed by upper bounds.
TEST(distance_node, backward_propagation)
{
  auto r = make_route({80.f, 600.f, 400.f},
                      {{0., DISTANCE_INF}, {0., 60.}, {0., DISTANCE_INF}, {0., DISTANCE_INF}},
                      /*max_cost=*/800.f);
  r.run_passes();

  EXPECT_DOUBLE_EQ(r.nodes[3].distance_backward, 0.);
  EXPECT_DOUBLE_EQ(r.nodes[2].distance_backward, 400.);
  EXPECT_DOUBLE_EQ(r.nodes[1].distance_backward, 1000.);
  EXPECT_DOUBLE_EQ(r.nodes[0].distance_backward, 1080.);

  EXPECT_DOUBLE_EQ(r.nodes[1].distance_window_backward, 60.);
  EXPECT_DOUBLE_EQ(r.nodes[1].excess_backward, 0.);
  EXPECT_DOUBLE_EQ(r.nodes[0].distance_window_backward, 0.);
  EXPECT_DOUBLE_EQ(r.nodes[0].excess_backward, 20.);
  EXPECT_DOUBLE_EQ(r.nodes[0].backward_excess(r.vehicle_info), 300.);
}

// combine() returns 0 at every split point of a window-feasible route.
TEST(distance_node, combine_invariant_feasible)
{
  auto r = make_route({50.f, 55.f, 100.f},
                      {{0., DISTANCE_INF}, {0., 60.}, {0., DISTANCE_INF}, {0., DISTANCE_INF}},
                      /*max_cost=*/1000.f);
  r.run_passes();

  for (size_t k = 0; k + 1 < r.nodes.size(); ++k) {
    double c = distance_node::combine(r.nodes[k], r.nodes[k + 1], r.vehicle_info, r.arcs[k]);
    EXPECT_DOUBLE_EQ(c, 0.) << "split (" << k << ", " << (k + 1) << ") got " << c;
  }
}

// combine() reports the same window-violation excess at every split point.
TEST(distance_node, combine_invariant_window_violation)
{
  auto r = make_route({80.f, 600.f, 400.f},
                      {{0., DISTANCE_INF}, {0., 60.}, {0., DISTANCE_INF}, {0., DISTANCE_INF}},
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

// combine() reports the max_cost overage at every split point of a window-free route.
TEST(distance_node, combine_invariant_max_cost_only)
{
  auto r =
    make_route({400.f, 300.f, 400.f},
               {{0., DISTANCE_INF}, {0., DISTANCE_INF}, {0., DISTANCE_INF}, {0., DISTANCE_INF}},
               /*max_cost=*/1000.f);
  r.run_passes();

  double reference = distance_node::combine(r.nodes[0], r.nodes[1], r.vehicle_info, r.arcs[0]);
  EXPECT_DOUBLE_EQ(reference, 100.);  // total 1100, max_cost 1000.
  for (size_t k = 1; k + 1 < r.nodes.size(); ++k) {
    double c = distance_node::combine(r.nodes[k], r.nodes[k + 1], r.vehicle_info, r.arcs[k]);
    EXPECT_DOUBLE_EQ(c, reference);
  }
}

// End-of-route boundary plus max_cost overage matches combine() at the first split.
TEST(distance_node, compute_cost_combine_consistency)
{
  auto r = make_route({80.f, 600.f, 400.f},
                      {{0., DISTANCE_INF}, {0., 60.}, {0., DISTANCE_INF}, {0., DISTANCE_INF}},
                      /*max_cost=*/800.f);
  r.run_passes();

  auto const& end_node = r.nodes.back();
  double boundary =
    std::max(0., end_node.distance_window_forward - end_node.distance_window_backward);
  double total_distance = end_node.distance_forward;
  double max_cost_excess =
    std::max(0., total_distance - static_cast<double>(r.vehicle_info.max_cost));
  double total = end_node.excess_forward + boundary + max_cost_excess;

  double combine_at_first =
    distance_node::combine(r.nodes[0], r.nodes[1], r.vehicle_info, r.arcs[0]);

  EXPECT_DOUBLE_EQ(total, combine_at_first);
}

// compute_cost must not read the soft-cost span unless distance windows are enabled.
TEST(distance_route, distance_break_cost_requires_distance_window)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto distance_forward = cuopt::device_copy(std::vector<double>{0.}, stream);
  rmm::device_uvector<double> result(1, stream);

  distance_route::view_t route;
  route.dim_info.has_distance_window     = false;
  route.dim_info.has_distance_break_cost = true;
  route.distance_forward =
    raft::device_span<double>{distance_forward.data(), distance_forward.size()};
  ASSERT_TRUE(route.distance_break_cost_forward.empty());
  EXPECT_EQ(distance_route::get_shared_size(1, route.dim_info), 2 * sizeof(double));

  compute_distance_route_cost<<<1, 1, 0, stream>>>(route, result.data());
  RAFT_CUDA_TRY(cudaGetLastError());

  auto host_result = cuopt::host_copy(result, stream);
  EXPECT_DOUBLE_EQ(host_result[0], 0.);
}

// get_cost() agrees with combine() at every split point of a route.
TEST(distance_node, get_cost_combine_consistency)
{
  auto r = make_route({80.f, 600.f, 400.f},
                      {{0., DISTANCE_INF}, {0., 60.}, {0., DISTANCE_INF}, {0., DISTANCE_INF}},
                      /*max_cost=*/800.f);
  r.run_passes();

  detail::cost_dimension_info_t dim_info;
  dim_info.has_max_constraint      = true;
  dim_info.has_distance_window     = true;
  dim_info.has_distance_break_cost = true;

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
    EXPECT_DOUBLE_EQ(obj_cost[objective_t::DISTANCE_BREAK_COST], 0.);
  }
}

// combine() = break-window excess + max_cost overage (additive accounting).
TEST(distance_node, combine_additive_break_and_max_cost)
{
  // Arc 100 to break B with window [0, 50] (cumulative 100 → excess 50), then arcs 20 and 10.
  // Route total = 130, max_cost = 120, so max_cost overage = 10.
  // Expected combine value at every split = 50 (break) + 10 (max_cost) = 60.
  auto r = make_route({100.f, 20.f, 10.f},
                      {{0., DISTANCE_INF}, {0., 50.}, {0., DISTANCE_INF}, {0., DISTANCE_INF}},
                      /*max_cost=*/120.f);
  r.run_passes();

  double reference = distance_node::combine(r.nodes[0], r.nodes[1], r.vehicle_info, r.arcs[0]);
  EXPECT_DOUBLE_EQ(reference, 60.);
  for (size_t k = 1; k + 1 < r.nodes.size(); ++k) {
    double c = distance_node::combine(r.nodes[k], r.nodes[k + 1], r.vehicle_info, r.arcs[k]);
    EXPECT_DOUBLE_EQ(c, reference) << "split (" << k << ", " << (k + 1) << ") = " << c;
  }
}

// depot=0, orders=1-2, optional break locations=3-4 (used in 5x5 tests)
// clang-format off
constexpr std::array<float, 9> cost_matrix_3x3 = {
  0, 1, 1,
  1, 0, 1,
  1, 1, 0,
};
constexpr std::array<float, 25> cost_matrix_5x5 = {
  0, 1, 1, 1, 1,
  1, 0, 1, 1, 1,
  1, 1, 0, 1, 1,
  1, 1, 1, 0, 1,
  1, 1, 1, 1, 0,
};
// clang-format on

// End-to-end smoke test: distance breaks solve on a trivial 3x3 matrix without break locations.
TEST(distance_breaks, default_case)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix = copy_array_to_device(cost_matrix_3x3, stream);
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

// Distance-break cost defaults to weight 1, including when another objective is configured;
// an explicit zero disables it.
TEST(distance_breaks, default_objective_weight)
{
  enum class objective_mode { DEFAULTS, OMIT_DISTANCE_BREAK_COST, DISABLE_DISTANCE_BREAK_COST };
  for (auto mode : {objective_mode::DEFAULTS,
                    objective_mode::OMIT_DISTANCE_BREAK_COST,
                    objective_mode::DISABLE_DISTANCE_BREAK_COST}) {
    raft::handle_t handle;
    auto stream = handle.get_stream();

    std::vector<float> cost_matrix      = {0.f, 1.f, 1.f, 1.f, 0.f, 1.f, 1.f, 1.f, 0.f};
    std::vector<int> order_locations    = {1};
    std::vector<int> break_locations    = {2};
    std::vector<objective_t> objectives = {objective_t::COST};
    std::vector<float> weights = {mode == objective_mode::OMIT_DISTANCE_BREAK_COST ? 2.f : 1.f};
    if (mode == objective_mode::DISABLE_DISTANCE_BREAK_COST) {
      objectives.push_back(objective_t::DISTANCE_BREAK_COST);
      weights.push_back(0.f);
    }

    auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
    auto v_order_locations = cuopt::device_copy(order_locations, stream);
    auto v_break_locations = cuopt::device_copy(break_locations, stream);
    auto v_objectives      = cuopt::device_copy(objectives, stream);
    auto v_weights         = cuopt::device_copy(weights, stream);

    cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 1, 1);
    data_model.add_cost_matrix(v_cost_matrix.data());
    data_model.set_order_locations(v_order_locations.data());
    data_model.add_distance_break(0, 10.f, 100.f, 0, v_break_locations.data(), 1);

    if (mode != objective_mode::DEFAULTS) {
      data_model.set_objective_function(v_objectives.data(), v_weights.data(), weights.size());
    }

    auto settings = cuopt::routing::solver_settings_t<int, float>{};
    settings.set_time_limit(10);

    auto solution = cuopt::routing::solve(data_model, settings);
    handle.sync_stream();

    ASSERT_EQ(solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
    auto const& objective_values = solution.get_objectives();
    EXPECT_DOUBLE_EQ(objective_values.at(objective_t::COST), 3.);
    if (mode == objective_mode::DISABLE_DISTANCE_BREAK_COST) {
      EXPECT_EQ(objective_values.count(objective_t::DISTANCE_BREAK_COST), 0u);
      EXPECT_DOUBLE_EQ(solution.get_total_objective(), 3.);
    } else {
      EXPECT_DOUBLE_EQ(objective_values.at(objective_t::DISTANCE_BREAK_COST), 8.);
      EXPECT_DOUBLE_EQ(solution.get_total_objective(),
                       mode == objective_mode::OMIT_DISTANCE_BREAK_COST ? 14. : 11.);
    }
  }
}

// Break locations restrict where the break can be inserted.
TEST(distance_breaks, with_break_locations)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::vector<int> order_locations = {1, 2};
  std::vector<int> break_locations = {3, 4};

  auto v_cost_matrix     = copy_array_to_device(cost_matrix_5x5, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_break_locations = cuopt::device_copy(break_locations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 5, 2, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.add_distance_break(
    0, 0.f, 2.f, 1, v_break_locations.data(), (int)v_break_locations.size());
  data_model.add_distance_break(
    1, 0.f, 2.f, 1, v_break_locations.data(), (int)v_break_locations.size());
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

// Stacking add_distance_break calls produces one break per cycle per vehicle.
TEST(distance_breaks, multi_cycle)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  // Two vehicles, each with two charge cycles: [0, 2) and [2, 4).
  std::vector<int> order_locations = {1, 2};

  auto v_cost_matrix     = copy_array_to_device(cost_matrix_5x5, stream);
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

// Solver chooses the longer route so the break lands inside [0, d_max].
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
  std::vector<int> order_locations = {1};
  std::vector<int> break_locations = {2};

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix_3, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_break_locations = cuopt::device_copy(break_locations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 1, 1);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.add_distance_break(0, 0.f, 60.f, 0, v_break_locations.data(), 1);

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

// A configured objective weight makes the solver prefer a route that reaches distance_min.
TEST(distance_breaks, early_arrival_objective)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  // clang-format off
  std::vector<float> cost_matrix_4 = {
    0,   50,  50,  1,
    50,  0,   10,  60,
    50,  10,  0,   60,
    1,   60,  60,  0,
  };
  // clang-format on
  std::vector<int> order_locations     = {1, 2};
  std::vector<int> break_locations     = {3};
  std::vector<objective_t> objectives  = {objective_t::COST, objective_t::DISTANCE_BREAK_COST};
  std::vector<float> objective_weights = {1.f, 100.f};

  auto v_cost_matrix       = cuopt::device_copy(cost_matrix_4, stream);
  auto v_order_locations   = cuopt::device_copy(order_locations, stream);
  auto v_break_locations   = cuopt::device_copy(break_locations, stream);
  auto v_objectives        = cuopt::device_copy(objectives, stream);
  auto v_objective_weights = cuopt::device_copy(objective_weights, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 4, 1, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.add_distance_break(0, 40.f, 200.f, 0, v_break_locations.data(), 1);
  data_model.set_objective_function(
    v_objectives.data(), v_objective_weights.data(), v_objective_weights.size());

  auto settings = cuopt::routing::solver_settings_t<int, float>{};
  settings.set_time_limit(10);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  EXPECT_DOUBLE_EQ(routing_solution.get_objectives().at(objective_t::DISTANCE_BREAK_COST), 0.);

  host_assignment_t<int> h(routing_solution);
  float cumulative = 0.f;
  int prev_loc     = 0;
  bool found_break = false;
  for (size_t i = 0; i < h.locations.size(); ++i) {
    int loc = h.locations[i];
    cumulative += cost_matrix_4[prev_loc * 4 + loc];
    if (static_cast<node_type_t>(h.node_types[i]) == node_type_t::BREAK) {
      found_break = true;
      EXPECT_GE(cumulative, 40.f - 1e-3f);
      EXPECT_LE(cumulative, 200.f + 1e-3f);
    }
    prev_loc = loc;
  }
  EXPECT_TRUE(found_break);
}

// Only vehicles configured with a distance break receive break nodes.
TEST(distance_breaks, mixed_fleet)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix = copy_array_to_device(cost_matrix_3x3, stream);
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
