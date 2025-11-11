/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/solve.hpp>
#include <map>
#include <routing/utilities/check_constraints.hpp>
#include <utilities/copy_helpers.hpp>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

using i_t = int;
using f_t = float;

TEST(heterogenous, capacities)
{
  int nlocations                   = 5;
  int norders                      = 4;
  int nvehicles                    = 16;
  std::vector<float> cost_matrix   = {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                                      1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0};
  std::vector<int> order_locations = {2, 4, 3, 1};

  std::vector<int> pickup_orders, delivery_orders;
  pickup_orders   = {0, 2};
  delivery_orders = {1, 3};

  std::vector<int> order_service_times = {10, 10, 10, 10};

  //
  std::vector<int> capacities(nvehicles, 6);
  // capacities[nvehicles-1] = 9;
  capacities[13] = 9;
  // = {6, 9};
  std::vector<int> demands = {5, -5, 8, -8};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);

  rmm::device_uvector<int> v_pickup_orders(0, stream), v_delivery_orders(0, stream);

  v_pickup_orders   = cuopt::device_copy(pickup_orders, stream);
  v_delivery_orders = cuopt::device_copy(delivery_orders, stream);

  auto v_order_service_times = cuopt::device_copy(order_service_times, stream);
  auto v_capacities          = cuopt::device_copy(capacities, stream);
  auto v_demands             = cuopt::device_copy(demands, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_pickup_delivery_pairs(v_pickup_orders.data(), v_delivery_orders.data());

  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());

  data_model.set_order_service_times(v_order_service_times.data());
  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(4);

  auto routing_solution = cuopt::routing::solve(data_model, settings);

  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);
}

TEST(heterogenous, service_times)
{
  int nlocations                   = 5;
  int norders                      = 4;
  int nvehicles                    = 16;
  std::vector<float> cost_matrix   = {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                                      1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0};
  std::vector<int> order_locations = {2, 4, 3, 1};

  std::vector<int> pickup_orders, delivery_orders;
  pickup_orders   = {0, 2};
  delivery_orders = {1, 3};

  std::map<int, std::vector<int>> order_service_times;
  for (int i = 0; i < nvehicles; ++i) {
    order_service_times[i] = std::vector<int>(norders, 10 * i + 1);
  }

  std::vector<int> capacities(nvehicles, 9);
  std::vector<int> demands = {5, -5, 8, -8};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);

  rmm::device_uvector<int> v_pickup_orders(0, stream), v_delivery_orders(0, stream);

  v_pickup_orders   = cuopt::device_copy(pickup_orders, stream);
  v_delivery_orders = cuopt::device_copy(delivery_orders, stream);

  auto v_capacities = cuopt::device_copy(capacities, stream);
  auto v_demands    = cuopt::device_copy(demands, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_pickup_delivery_pairs(v_pickup_orders.data(), v_delivery_orders.data());

  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());

  std::map<int, rmm::device_uvector<int>> v_order_service_times;
  for (int v = 0; v < nvehicles; ++v) {
    v_order_service_times.emplace(v, cuopt::device_copy(order_service_times[v], stream));
    data_model.set_order_service_times(v_order_service_times.at(v).data(), v);
  }

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(4);

  auto routing_solution = cuopt::routing::solve(data_model, settings);

  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);
}

TEST(heterogenous, vehicle_tw)
{
  int nlocations = 10;
  int norders    = 4;
  int nvehicles  = 5;

  std::vector<float> cost_matrix(nlocations * nlocations, 1);
  for (int i = 0; i < nlocations; ++i) {
    cost_matrix[i + i * nlocations] = 0;
  }

  std::vector<int> order_locations = {2, 4, 3, 1};

  std::vector<int> pickup_orders, delivery_orders;
  pickup_orders   = {0, 2};
  delivery_orders = {1, 3};

  std::vector<int> order_service_times(norders, 10);

  // setup problem so that only vehicles 2 and 4 can serve the orders
  std::vector<int> vehicle_earliest_times = {10, 20, 30, 40, 50};
  std::vector<int> vehicle_latest_times   = {11, 21, 60, 41, 80};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);

  auto v_pickup_orders   = cuopt::device_copy(pickup_orders, stream);
  auto v_delivery_orders = cuopt::device_copy(delivery_orders, stream);

  auto v_order_service_times = cuopt::device_copy(order_service_times, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_pickup_delivery_pairs(v_pickup_orders.data(), v_delivery_orders.data());

  data_model.set_order_service_times(v_order_service_times.data());

  auto v_vehicle_earliest_times = cuopt::device_copy(vehicle_earliest_times, stream);
  auto v_vehicle_latest_times   = cuopt::device_copy(vehicle_latest_times, stream);
  data_model.set_vehicle_time_windows(v_vehicle_earliest_times.data(),
                                      v_vehicle_latest_times.data());

  auto settings = cuopt::routing::solver_settings_t<int, float>{};
  settings.set_time_limit(2);
  auto routing_solution = cuopt::routing::solve(data_model, settings);

  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  // host_route.print();

  auto const& truck_ids = host_route.truck_id;
  for (auto& truck_id : truck_ids) {
    EXPECT_EQ(truck_id == 2 || truck_id == 4, true);
  }
  check_route(data_model, host_route);
}

TEST(heterogenous, order_vehicle_match)
{
  int nlocations = 10;
  int norders    = 4;
  int nvehicles  = 5;

  std::vector<float> cost_matrix(nlocations * nlocations, 10);
  for (int i = 0; i < nlocations; ++i) {
    cost_matrix[i + i * nlocations] = 0;
  }

  std::vector<int> order_locations = {2, 4, 3, 1};

  std::vector<int> pickup_orders, delivery_orders;
  pickup_orders   = {0, 2};
  delivery_orders = {1, 3};

  // Only vehicle 1 and 3 can serve these orders
  std::unordered_map<i_t, std::vector<i_t>> order_vehicle_match{
    {0, std::vector{1}},
    {1, std::vector{1}},
    {2, std::vector{3}},
    {3, std::vector{3}},
  };

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);

  auto v_pickup_orders   = cuopt::device_copy(pickup_orders, stream);
  auto v_delivery_orders = cuopt::device_copy(delivery_orders, stream);

  std::unordered_map<i_t, rmm::device_uvector<i_t>> order_vehicle_match_d;
  for (const auto& [order, vehicles] : order_vehicle_match) {
    order_vehicle_match_d.emplace(order, cuopt::device_copy(vehicles, handle.get_stream()));
  }

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_pickup_delivery_pairs(v_pickup_orders.data(), v_delivery_orders.data());

  for (const auto& [order, vehicles] : order_vehicle_match_d) {
    data_model.add_order_vehicle_match(order, vehicles.data(), vehicles.size());
  }

  auto routing_solution = cuopt::routing::solve(data_model);

  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  // host_route.print();

  check_route(data_model, host_route);

  auto const& node_types = host_route.node_types;
  auto const& order_ids  = host_route.route;
  auto const& truck_ids  = host_route.truck_id;
  for (size_t i = 0; i < order_ids.size(); ++i) {
    if (node_types[i] != (int)node_type_t::DEPOT) {
      int order_id = order_ids[i];
      if (order_id == 0 || order_id == 1) {
        EXPECT_EQ(truck_ids[i], 1);
      } else if (order_id == 2 || order_id == 3) {
        EXPECT_EQ(truck_ids[i], 3);
      }
    }
  }
}

TEST(heterogenous, multi_cost)
{
  int nlocations = 10;
  int norders    = 4;
  int nvehicles  = 2;

  std::vector<uint8_t> vehicle_types{1, 2};
  std::vector<float> cost_matrix_1(nlocations * nlocations, 10);
  std::vector<float> cost_matrix_2(nlocations * nlocations, 100);
  for (int i = 0; i < nlocations; ++i) {
    cost_matrix_1[i + i * nlocations] = 0;
    cost_matrix_2[i + i * nlocations] = 0;
  }

  std::vector<int> order_locations = {2, 4, 3, 1};

  std::vector<int> pickup_orders, delivery_orders;
  pickup_orders   = {0, 2};
  delivery_orders = {1, 3};

  // only first vehicle can serve both orders
  std::vector<int> order_earliest = {0, 0, 0, 0};
  std::vector<int> order_latest   = {330, 330, 330, 330};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_vehicle_types   = cuopt::device_copy(vehicle_types, stream);
  auto v_cost_matrix_1   = cuopt::device_copy(cost_matrix_1, stream);
  auto v_cost_matrix_2   = cuopt::device_copy(cost_matrix_2, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);

  auto v_pickup_orders   = cuopt::device_copy(pickup_orders, stream);
  auto v_delivery_orders = cuopt::device_copy(delivery_orders, stream);

  auto v_order_earliest = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest   = cuopt::device_copy(order_latest, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.set_vehicle_types(v_vehicle_types.data());
  data_model.add_cost_matrix(v_cost_matrix_1.data(), 1);
  data_model.add_cost_matrix(v_cost_matrix_2.data(), 2);
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_pickup_delivery_pairs(v_pickup_orders.data(), v_delivery_orders.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(7);

  auto routing_solution = cuopt::routing::solve(data_model, settings);

  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  // host_route.print();

  check_route(data_model, host_route);

  // only first truck should be able to handle
  auto const& truck_ids = host_route.truck_id;
  for (auto& truck : truck_ids) {
    EXPECT_EQ(truck, 0);
  }
}

TEST(heterogenous, vehicle_locations)
{
  int nlocations = 10;
  int norders    = 4;
  int nvehicles  = 4;

  std::vector<float> cost_matrix(nlocations * nlocations, 10);
  for (int i = 0; i < nlocations; ++i) {
    cost_matrix[i + i * nlocations] = 0;
  }

  std::vector<int> order_locations = {2, 4, 3, 1};

  std::vector<int> pickup_orders   = {0, 2};
  std::vector<int> delivery_orders = {1, 3};

  std::vector<int> vehicle_locations  = {6, 7, 8, 9};
  std::vector<bool> drop_return_trips = {0, 0, 1, 1};
  std::vector<bool> skip_first_trips  = {0, 1, 0, 1};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);

  auto v_pickup_orders   = cuopt::device_copy(pickup_orders, stream);
  auto v_delivery_orders = cuopt::device_copy(delivery_orders, stream);

  auto v_vehicle_locations = cuopt::device_copy(vehicle_locations, stream);
  auto v_drop_return_trips = cuopt::device_copy(drop_return_trips, stream);
  auto v_skip_first_trips  = cuopt::device_copy(skip_first_trips, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());

  data_model.set_vehicle_locations(v_vehicle_locations.data(), v_vehicle_locations.data());
  data_model.set_drop_return_trips(v_drop_return_trips.data());
  data_model.set_skip_first_trips(v_skip_first_trips.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_pickup_delivery_pairs(v_pickup_orders.data(), v_delivery_orders.data());

  cuopt::routing::solver_settings_t<i_t, f_t> settings;
  settings.set_time_limit(5);

  auto routing_solution = cuopt::routing::solve(data_model, settings);

  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  // host_route.print();

  check_route(data_model, host_route);

  ASSERT_EQ(routing_solution.get_vehicle_count(), 1);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
