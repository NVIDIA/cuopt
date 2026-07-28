/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/solve.hpp>
#include <utilities/copy_helpers.hpp>

#include <numeric>
#include <vector>

namespace cuopt::routing::test {

// Covers capacity_route_t::resize preserving the row-major layout of a multi-dimension
// capacity route: many orders over a few vehicles force a route to grow past its base
// allocation, and a second capacity dimension exposes the resize. See issue #1618.
TEST(capacity_route_resize, multi_dim_grow_preserves_second_dimension)
{
  constexpr int norders    = 130;
  constexpr int nvehicles  = 3;
  constexpr int nloops     = 3;
  constexpr int time_limit = 2;
  constexpr int nlocations = norders + 1;
  constexpr int cap        = 1000000;  // loose: never binding, so any excess is corruption

  // Locations on a line; cost = |a - b| so consolidating orders onto a route is cheap.
  std::vector<float> cost_matrix(nlocations * nlocations);
  for (int a = 0; a < nlocations; ++a) {
    for (int b = 0; b < nlocations; ++b) {
      cost_matrix[a * nlocations + b] = std::abs(a - b);
    }
  }
  std::vector<int> order_locations(norders);
  std::iota(order_locations.begin(), order_locations.end(), 1);  // order i -> location i+1
  std::vector<int> vehicle_start(nvehicles, 0), vehicle_return(nvehicles, 0);
  std::vector<int> demand_weight(norders, 1);
  std::vector<int> demand_volume(norders, 1);
  std::vector<int> capacity_weight(nvehicles, cap);
  std::vector<int> capacity_volume(nvehicles, cap);

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = device_copy(cost_matrix, stream);
  auto v_order_locations = device_copy(order_locations, stream);
  auto v_start           = device_copy(vehicle_start, stream);
  auto v_return          = device_copy(vehicle_return, stream);
  auto v_demand_weight   = device_copy(demand_weight, stream);
  auto v_demand_volume   = device_copy(demand_volume, stream);
  auto v_cap_weight      = device_copy(capacity_weight, stream);
  auto v_cap_volume      = device_copy(capacity_volume, stream);

  data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_vehicle_locations(v_start.data(), v_return.data());
  data_model.add_capacity_dimension("weight", v_demand_weight.data(), v_cap_weight.data());
  data_model.add_capacity_dimension("volume", v_demand_volume.data(), v_cap_volume.data());

  solver_settings_t<int, float> settings;
  settings.set_time_limit(time_limit);

  for (int loop = 0; loop < nloops; ++loop) {
    auto routing_solution = solve(data_model, settings);
    handle.sync_stream();
    ASSERT_EQ(routing_solution.get_status(), solution_status_t::SUCCESS) << "loop " << loop;

    // The in-solver feasibility assert is the primary detector (it aborts during solve on
    // the corrupted dimension). These host-side checks are a secondary guard for release
    // builds where asserts are compiled out: a valid solution serves every order exactly
    // once and respects each configured capacity.
    host_assignment_t<int> host_route(routing_solution);
    std::vector<int> load_weight(nvehicles, 0), load_volume(nvehicles, 0);
    std::vector<int> times_served(norders, 0);
    for (size_t i = 0; i < host_route.route.size(); ++i) {
      auto node_type = static_cast<node_type_t>(host_route.node_types[i]);
      if (node_type == node_type_t::DEPOT || node_type == node_type_t::BREAK) { continue; }
      int truck = host_route.truck_id[i];
      int order = host_route.route[i];
      ++times_served[order];
      load_weight[truck] += demand_weight[order];
      load_volume[truck] += demand_volume[order];
    }
    for (int order = 0; order < norders; ++order) {
      ASSERT_EQ(times_served[order], 1)
        << "order " << order << " served " << times_served[order] << " times, loop " << loop;
    }
    for (int v = 0; v < nvehicles; ++v) {
      ASSERT_LE(load_weight[v], capacity_weight[v])
        << "weight cap on veh " << v << " loop " << loop;
      ASSERT_LE(load_volume[v], capacity_volume[v])
        << "volume cap on veh " << v << " loop " << loop;
    }
  }
}

}  // namespace cuopt::routing::test
