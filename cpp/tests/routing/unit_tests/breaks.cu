/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <routing/utilities/check_constraints.hpp>
#include <routing/utilities/test_utilities.hpp>

#include <cuopt/routing/solve.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

static std::vector<float> cost_matrix   = {0, 1, 1, 1, 0, 1, 1, 1, 0};
static std::vector<int> break_earliest  = {0, 1};
static std::vector<int> break_latest    = {2, 3};
static std::vector<int> break_service   = {1, 1};
static std::vector<int> break_locations = {1, 2};

TEST(vehicle_breaks, default_case)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_break_earliest  = cuopt::device_copy(break_earliest, stream);
  auto v_break_latest    = cuopt::device_copy(break_latest, stream);
  auto v_break_service   = cuopt::device_copy(break_service, stream);
  auto v_break_locations = cuopt::device_copy(break_locations, stream);
  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_cost_matrix.data());
  data_model.add_break_dimension(
    v_break_earliest.data(), v_break_latest.data(), v_break_service.data());
  data_model.set_break_locations(v_break_locations.data(), v_break_locations.size());
  data_model.set_min_vehicles(2);

  auto routing_solution = cuopt::routing::solve(data_model);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  ASSERT_LT(abs(routing_solution.get_total_objective() - 4.0f), 0.001);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);
}

TEST(vehicle_breaks, non_default_case)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_break_earliest  = cuopt::device_copy(break_earliest, stream);
  auto v_break_latest    = cuopt::device_copy(break_latest, stream);
  auto v_break_service   = cuopt::device_copy(break_service, stream);
  auto v_break_locations = cuopt::device_copy(break_locations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_cost_matrix.data());
  data_model.add_break_dimension(
    v_break_earliest.data(), v_break_latest.data(), v_break_service.data());
  data_model.set_break_locations(v_break_locations.data(), v_break_locations.size());
  data_model.set_min_vehicles(2);

  auto settings = cuopt::routing::solver_settings_t<int, float>{};
  settings.set_time_limit(2);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  ASSERT_LT(abs(routing_solution.get_total_objective() - 4.0f), 0.001);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);
}

TEST(vehicle_breaks, only_one_feasible_vehicle)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  int nlocations = 3;
  int norders    = 1;
  int nvehicles  = 10;

  std::vector<float> cost_matrix   = {0., 1., 1., 1., 0., 1., 1., 1., 0.};
  std::vector<int> order_locations = {2};
  std::vector<int> order_earliest  = {780};
  std::vector<int> order_latest    = {1000};
  std::vector<int> order_service   = {420};

  std::vector<int> break_earliest(nvehicles, 1020);
  std::vector<int> break_latest(nvehicles, 1080);
  std::vector<int> break_duration(nvehicles, 0);

  // Make sure just one vehicle can serve
  int vehicle_that_can_serve             = (int)(0.7 * nvehicles);
  break_earliest[vehicle_that_can_serve] = 0;
  break_latest[vehicle_that_can_serve]   = 1440;

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_order_earliest  = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest    = cuopt::device_copy(order_latest, stream);
  auto v_order_service   = cuopt::device_copy(order_service, stream);

  auto v_break_earliest = cuopt::device_copy(break_earliest, stream);
  auto v_break_latest   = cuopt::device_copy(break_latest, stream);
  auto v_break_duration = cuopt::device_copy(break_duration, stream);

  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_order_service_times(v_order_service.data());

  data_model.add_break_dimension(
    v_break_earliest.data(), v_break_latest.data(), v_break_duration.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(10);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);

  for (auto& vehicle_id : h_routing_solution.truck_id) {
    ASSERT_EQ(vehicle_id, vehicle_that_can_serve);
  }
}

TEST(vehicle_breaks, vehicle_time_windows)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  int nlocations = 4;
  int norders    = 1;
  int nvehicles  = 24;

  std::vector<float> cost_matrix = {
    0., 10., 10., 10., 10., 0., 10., 10., 10., 10., 0., 10., 10., 10., 10., 0.};

  std::vector<float> time_matrix = {
    0., 15., 15., 15., 15., 0., 15., 15., 15., 15., 0., 15., 15., 15., 15., 0.};

  std::vector<int> order_vehicle_match = {0, 14, 9, 17, 22, 23, 1, 8};

  std::vector<int> order_locations = {0};
  std::vector<int> order_earliest  = {930};
  std::vector<int> order_latest    = {1080};
  std::vector<int> order_service   = {180};

  std::vector<int> vehicle_start_locations(nvehicles, 3);
  std::vector<int> vehicle_return_locations(nvehicles, 3);

  std::vector<int> vehicle_earliest(nvehicles, 735);
  std::vector<int> vehicle_latest(nvehicles, 1260);
  vehicle_earliest[nvehicles - 1] = 0;
  vehicle_latest[nvehicles - 1]   = 30001;

  std::vector<int> break_earliest(nvehicles, 1020);
  std::vector<int> break_latest(nvehicles, 1080);
  std::vector<int> break_duration(nvehicles, 60);

  break_earliest[nvehicles - 1] = 1;
  break_latest[nvehicles - 1]   = 3000;
  break_duration[nvehicles - 1] = 0;

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);

  auto v_cost_matrix = cuopt::device_copy(cost_matrix, stream);
  auto v_time_matrix = cuopt::device_copy(time_matrix, stream);

  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_order_earliest  = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest    = cuopt::device_copy(order_latest, stream);
  auto v_order_service   = cuopt::device_copy(order_service, stream);

  auto v_vehicle_earliest         = cuopt::device_copy(vehicle_earliest, stream);
  auto v_vehicle_latest           = cuopt::device_copy(vehicle_latest, stream);
  auto v_vehicle_start_locations  = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_vehicle_return_locations = cuopt::device_copy(vehicle_return_locations, stream);

  auto v_order_vehicle_match = cuopt::device_copy(order_vehicle_match, stream);

  auto v_break_earliest = cuopt::device_copy(break_earliest, stream);
  auto v_break_latest   = cuopt::device_copy(break_latest, stream);
  auto v_break_duration = cuopt::device_copy(break_duration, stream);

  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_time_matrix.data());

  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_order_service_times(v_order_service.data());

  data_model.set_vehicle_locations(v_vehicle_start_locations.data(),
                                   v_vehicle_return_locations.data());
  data_model.set_vehicle_time_windows(v_vehicle_earliest.data(), v_vehicle_latest.data());

  data_model.add_order_vehicle_match(
    0, v_order_vehicle_match.data(), (int)v_order_vehicle_match.size());

  data_model.add_break_dimension(
    v_break_earliest.data(), v_break_latest.data(), v_break_duration.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(10);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();
  host_assignment_t<int> h_routing_solution(routing_solution);

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  check_route(data_model, h_routing_solution);

  // h_routing_solution.print();

  for (auto& vehicle_id : h_routing_solution.truck_id) {
    ASSERT_EQ(vehicle_id, nvehicles - 1);
  }
}

// Test uniform breaks (Solomon 100 nodes)
TEST(vehicle_breaks, uniform_breaks)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::string path = cuopt::test::get_rapids_dataset_root_dir() + "/solomon/In/r107.txt";
  Route<int, float> route;
  load_solomon(path, route, 101);

  int nodes       = route.n_locations;
  int n_orders    = nodes - 1;
  int vehicle_num = 25;

  std::vector<float> cost_matrix(nodes * nodes);
  build_dense_matrix(cost_matrix.data(), route.x_h, route.y_h);

  std::vector<int> order_locations(n_orders), order_earliest(n_orders), order_latest(n_orders),
    order_service(n_orders);
  for (int i = 0; i < n_orders; ++i) {
    order_locations[i] = i + 1;
    order_earliest[i]  = route.earliest_time_h[i + 1];
    order_latest[i]    = route.latest_time_h[i + 1];
    order_service[i]   = route.service_time_h[i + 1];
  }
  int num_breaks = 2;
  std::vector<int> break_earliest(vehicle_num * num_breaks);
  std::vector<int> break_latest(vehicle_num * num_breaks);
  std::vector<int> break_duration(vehicle_num * num_breaks);
  for (int v = 0; v < vehicle_num; ++v) {
    break_earliest[v * num_breaks + 0] = 40;
    break_latest[v * num_breaks + 0]   = 50;
    break_duration[v * num_breaks + 0] = 10;
    break_earliest[v * num_breaks + 1] = 170;
    break_latest[v * num_breaks + 1]   = 180;
    break_duration[v * num_breaks + 1] = 10;
  }

  std::vector<int> break_locations(nodes);
  for (int i = 0; i < nodes; ++i) {
    break_locations[i] = i;
  }

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nodes, vehicle_num, n_orders);

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_order_earliest  = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest    = cuopt::device_copy(order_latest, stream);
  auto v_order_service   = cuopt::device_copy(order_service, stream);
  auto v_break_locations = cuopt::device_copy(break_locations, stream);

  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_order_service_times(v_order_service.data());
  data_model.set_break_locations(v_break_locations.data(), v_break_locations.size());

  std::vector<int> dim0_earliest(vehicle_num), dim0_latest(vehicle_num), dim0_duration(vehicle_num);
  std::vector<int> dim1_earliest(vehicle_num), dim1_latest(vehicle_num), dim1_duration(vehicle_num);
  for (int v = 0; v < vehicle_num; ++v) {
    dim0_earliest[v] = break_earliest[v * num_breaks + 0];
    dim0_latest[v]   = break_latest[v * num_breaks + 0];
    dim0_duration[v] = break_duration[v * num_breaks + 0];
    dim1_earliest[v] = break_earliest[v * num_breaks + 1];
    dim1_latest[v]   = break_latest[v * num_breaks + 1];
    dim1_duration[v] = break_duration[v * num_breaks + 1];
  }
  auto v_break_earliest_0 = cuopt::device_copy(dim0_earliest, stream);
  auto v_break_latest_0   = cuopt::device_copy(dim0_latest, stream);
  auto v_break_duration_0 = cuopt::device_copy(dim0_duration, stream);
  data_model.add_break_dimension(
    v_break_earliest_0.data(), v_break_latest_0.data(), v_break_duration_0.data());

  auto v_break_earliest_1 = cuopt::device_copy(dim1_earliest, stream);
  auto v_break_latest_1   = cuopt::device_copy(dim1_latest, stream);
  auto v_break_duration_1 = cuopt::device_copy(dim1_duration, stream);
  data_model.add_break_dimension(
    v_break_earliest_1.data(), v_break_latest_1.data(), v_break_duration_1.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(30);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);
}

// Test non-uniform breaks (Solomon 100 nodes)
TEST(vehicle_breaks, non_uniform_breaks)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::string path = cuopt::test::get_rapids_dataset_root_dir() + "/solomon/In/r107.txt";
  Route<int, float> route;
  load_solomon(path, route, 101);

  int nodes       = route.n_locations;
  int n_orders    = nodes - 1;
  int vehicle_num = 30;

  std::vector<float> cost_matrix(nodes * nodes);
  build_dense_matrix(cost_matrix.data(), route.x_h, route.y_h);

  std::vector<int> order_locations(n_orders), order_earliest(n_orders), order_latest(n_orders),
    order_service(n_orders);
  for (int i = 0; i < n_orders; ++i) {
    order_locations[i] = i + 1;
    order_earliest[i]  = route.earliest_time_h[i + 1];
    order_latest[i]    = route.latest_time_h[i + 1];
    order_service[i]   = route.service_time_h[i + 1];
  }
  int num_v_type_1 = vehicle_num / 2;
  int num_breaks   = 3;

  // Type 1: [40,50]/5, [100,120]/20, [170,180]/10
  // Type 2: [60,90]/20, [110,120]/10, [200,210]/5
  std::vector<int> break_earliest(vehicle_num * num_breaks);
  std::vector<int> break_latest(vehicle_num * num_breaks);
  std::vector<int> break_duration(vehicle_num * num_breaks);
  for (int v = 0; v < num_v_type_1; ++v) {
    break_earliest[v * num_breaks + 0] = 40;
    break_latest[v * num_breaks + 0]   = 50;
    break_duration[v * num_breaks + 0] = 5;
    break_earliest[v * num_breaks + 1] = 100;
    break_latest[v * num_breaks + 1]   = 120;
    break_duration[v * num_breaks + 1] = 20;
    break_earliest[v * num_breaks + 2] = 170;
    break_latest[v * num_breaks + 2]   = 180;
    break_duration[v * num_breaks + 2] = 10;
  }
  for (int v = num_v_type_1; v < vehicle_num; ++v) {
    break_earliest[v * num_breaks + 0] = 60;
    break_latest[v * num_breaks + 0]   = 90;
    break_duration[v * num_breaks + 0] = 20;
    break_earliest[v * num_breaks + 1] = 110;
    break_latest[v * num_breaks + 1]   = 120;
    break_duration[v * num_breaks + 1] = 10;
    break_earliest[v * num_breaks + 2] = 200;
    break_latest[v * num_breaks + 2]   = 210;
    break_duration[v * num_breaks + 2] = 5;
  }

  // Depot (0) excluded from break locations
  std::vector<int> break_locations(nodes - 1);
  for (int i = 0; i < nodes - 1; ++i) {
    break_locations[i] = i + 1;
  }

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nodes, vehicle_num, n_orders);

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_order_earliest  = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest    = cuopt::device_copy(order_latest, stream);
  auto v_order_service   = cuopt::device_copy(order_service, stream);
  auto v_break_locations = cuopt::device_copy(break_locations, stream);

  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_order_service_times(v_order_service.data());
  data_model.set_break_locations(v_break_locations.data(), v_break_locations.size());

  for (int b = 0; b < num_breaks; ++b) {
    std::vector<int> e(vehicle_num), l(vehicle_num), d(vehicle_num);
    for (int v = 0; v < vehicle_num; ++v) {
      e[v] = break_earliest[v * num_breaks + b];
      l[v] = break_latest[v * num_breaks + b];
      d[v] = break_duration[v * num_breaks + b];
    }
    auto v_e = cuopt::device_copy(e, stream);
    auto v_l = cuopt::device_copy(l, stream);
    auto v_d = cuopt::device_copy(d, stream);
    data_model.add_break_dimension(v_e.data(), v_l.data(), v_d.data());
  }

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(30);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  host_assignment_t<int> h_routing_solution(routing_solution);
  check_route(data_model, h_routing_solution);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
