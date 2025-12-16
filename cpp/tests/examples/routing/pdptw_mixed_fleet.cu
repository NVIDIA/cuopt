/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/solve.hpp>
#include <utilities/copy_helpers.hpp>
#include "../../routing/utilities/check_constraints.hpp"

#include <tuple>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

bool pdptw_mixed_fleet_basic_mixed_fleet_scenario()
{
  // 5 locations: depot (0) and 4 delivery points
  int nlocations = 5;
  int nvehicles  = 3;  // 2 small trucks + 1 large truck
  int norders    = 4;  // 2 pickup-delivery pairs

  // Cost matrix representing distances between locations
  std::vector<float> cost_matrix = {
    0,  10, 15, 20, 25,  // depot to delivery points
    10, 0,  35, 25, 30,  // delivery point 1 to others
    15, 35, 0,  30, 20,  // delivery point 2 to others
    20, 25, 30, 0,  15,  // delivery point 3 to others
    25, 30, 20, 15, 0    // delivery point 4 to others
  };

  // Vehicle types: 0 for small truck, 1 for large truck
  std::vector<uint8_t> vehicle_types = {0, 0, 1};

  // Order locations (pickup and delivery points)
  std::vector<int> order_locations = {1, 2, 3, 4};

  // Pickup and delivery pairs
  std::vector<int> pickup_orders   = {0, 2};
  std::vector<int> delivery_orders = {1, 3};

  // Time windows for orders (earliest and latest times)
  std::vector<int> order_earliest = {0, 0, 0, 0};
  std::vector<int> order_latest   = {100, 100, 100, 100};

  // Service times at each location
  std::vector<int> order_service_times = {10, 10, 10, 10};

  // Vehicle capacities (small trucks: 100, large truck: 200)
  std::vector<int> capacities = {100, 100, 200};

  // Delivery demands (positive for pickup, negative for delivery)
  std::vector<int> demands = {50, -50, 150, -150};

  // Vehicle start and end locations (all at depot)
  std::vector<int> vehicle_start_locations = {0, 0, 0};
  std::vector<int> vehicle_end_locations   = {0, 0, 0};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  // Copy data to device
  auto v_cost_matrix         = cuopt::device_copy(cost_matrix, stream);
  auto v_vehicle_types       = cuopt::device_copy(vehicle_types, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_pickup_orders       = cuopt::device_copy(pickup_orders, stream);
  auto v_delivery_orders     = cuopt::device_copy(delivery_orders, stream);
  auto v_order_earliest      = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest        = cuopt::device_copy(order_latest, stream);
  auto v_order_service_times = cuopt::device_copy(order_service_times, stream);
  auto v_capacities          = cuopt::device_copy(capacities, stream);
  auto v_demands             = cuopt::device_copy(demands, stream);
  auto v_start_locations     = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_end_locations       = cuopt::device_copy(vehicle_end_locations, stream);

  // Create and configure the data model
  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);

  // Set vehicle types
  data_model.set_vehicle_types(v_vehicle_types.data());

  // Add cost matrix
  data_model.add_cost_matrix(v_cost_matrix.data());

  // Set order locations and pickup-delivery pairs
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_pickup_delivery_pairs(v_pickup_orders.data(), v_delivery_orders.data());

  // Set time windows and service times
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_order_service_times(v_order_service_times.data());

  // Set vehicle locations
  data_model.set_vehicle_locations(v_start_locations.data(), v_end_locations.data());

  // Add capacity constraints
  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());

  // Configure solver settings
  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(10);  // 10 second time limit

  // Solve the problem
  auto routing_solution = cuopt::routing::solve(data_model, settings);

  handle.sync_stream();
  if (routing_solution.get_status() != cuopt::routing::solution_status_t::SUCCESS) { return false; }

  // Get and verify the solution
  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);

  // Verify that the large truck (type 1) is used for the larger demand
  auto const& truck_ids = host_route.truck_id;
  auto const& order_ids = host_route.route;
  for (size_t i = 0; i < order_ids.size(); ++i) {
    if (order_ids[i] == 2 || order_ids[i] == 3) {  // Larger demand pair
      if (vehicle_types[truck_ids[i]] != 1) {      // Should be assigned to large truck
        return false;
      }
    }
  }
  return true;
}

bool pdptw_mixed_fleet_complex_mixed_fleet_scenario()
{
  // 8 locations: depot (0) and 7 delivery points
  int nlocations = 8;
  int nvehicles  = 4;  // 2 small trucks + 2 large trucks
  int norders    = 6;  // 3 pickup-delivery pairs

  // Cost matrix representing distances between locations
  std::vector<float> cost_matrix = {
    0,  10, 15, 20, 25, 30, 35, 40,  // depot to delivery points
    10, 0,  35, 25, 30, 20, 25, 30,  // delivery point 1 to others
    15, 35, 0,  30, 20, 25, 30, 35,  // delivery point 2 to others
    20, 25, 30, 0,  15, 30, 35, 40,  // delivery point 3 to others
    25, 30, 20, 15, 0,  35, 40, 45,  // delivery point 4 to others
    30, 20, 25, 30, 35, 0,  45, 50,  // delivery point 5 to others
    35, 25, 30, 35, 40, 45, 0,  55,  // delivery point 6 to others
    40, 30, 35, 40, 45, 50, 55, 0    // delivery point 7 to others
  };

  // Vehicle types: 0 for small truck, 1 for large truck
  std::vector<uint8_t> vehicle_types = {0, 0, 1, 1};

  // Order locations (pickup and delivery points)
  std::vector<int> order_locations = {1, 2, 3, 4, 5, 6};

  // Pickup and delivery pairs
  std::vector<int> pickup_orders   = {0, 2, 4};
  std::vector<int> delivery_orders = {1, 3, 5};

  // Time windows for orders (earliest and latest times)
  std::vector<int> order_earliest = {0, 0, 0, 0, 0, 0};
  std::vector<int> order_latest   = {200, 200, 200, 200, 200, 200};

  // Service times at each location
  std::vector<int> order_service_times = {15, 15, 20, 20, 25, 25};

  // Vehicle capacities (small trucks: 100, large trucks: 200)
  std::vector<int> capacities = {100, 100, 200, 200};

  // Delivery demands (positive for pickup, negative for delivery)
  std::vector<int> demands = {50, -50, 150, -150, 100, -100};

  // Vehicle start and end locations (all at depot)
  std::vector<int> vehicle_start_locations = {0, 0, 0, 0};
  std::vector<int> vehicle_end_locations   = {0, 0, 0, 0};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  // Copy data to device
  auto v_cost_matrix         = cuopt::device_copy(cost_matrix, stream);
  auto v_vehicle_types       = cuopt::device_copy(vehicle_types, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_pickup_orders       = cuopt::device_copy(pickup_orders, stream);
  auto v_delivery_orders     = cuopt::device_copy(delivery_orders, stream);
  auto v_order_earliest      = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest        = cuopt::device_copy(order_latest, stream);
  auto v_order_service_times = cuopt::device_copy(order_service_times, stream);
  auto v_capacities          = cuopt::device_copy(capacities, stream);
  auto v_demands             = cuopt::device_copy(demands, stream);
  auto v_start_locations     = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_end_locations       = cuopt::device_copy(vehicle_end_locations, stream);

  // Create and configure the data model
  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);

  // Set vehicle types
  data_model.set_vehicle_types(v_vehicle_types.data());

  // Add cost matrix
  data_model.add_cost_matrix(v_cost_matrix.data());

  // Set order locations and pickup-delivery pairs
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_pickup_delivery_pairs(v_pickup_orders.data(), v_delivery_orders.data());

  // Set time windows and service times
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_order_service_times(v_order_service_times.data());

  // Set vehicle locations
  data_model.set_vehicle_locations(v_start_locations.data(), v_end_locations.data());

  // Add capacity constraints
  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());

  // Configure solver settings
  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(15);  // 15 second time limit

  // Solve the problem
  auto routing_solution = cuopt::routing::solve(data_model, settings);

  handle.sync_stream();
  if (routing_solution.get_status() != cuopt::routing::solution_status_t::SUCCESS) { return false; }

  // Get and verify the solution
  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);

  // Verify that the large trucks (type 1) are used for the larger demands
  auto const& truck_ids = host_route.truck_id;
  auto const& order_ids = host_route.route;
  for (size_t i = 0; i < order_ids.size(); ++i) {
    if (order_ids[i] == 2 || order_ids[i] == 3) {  // Largest demand pair (150)
      if (vehicle_types[truck_ids[i]] != 1) {      // Should be assigned to large truck
        return false;
      }
    }
  }
  return true;
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt

int main()
{
  std::vector<std::string> example_names = {"pdptw_mixed_fleet_basic_mixed_fleet_scenario",
                                            "pdptw_mixed_fleet_complex_mixed_fleet_scenario"};

  std::vector<std::function<bool()>> example_functions = {
    cuopt::routing::test::pdptw_mixed_fleet_basic_mixed_fleet_scenario,
    cuopt::routing::test::pdptw_mixed_fleet_complex_mixed_fleet_scenario};

  int exit_code = 0;
  for (size_t i = 0; i < example_names.size(); ++i) {
    int status = example_functions[i]();
    std::cout << "Example: " << example_names[i]
              << " - Status: " << (status ? "Success" : "Failure") << std::endl;
    if (!status) { exit_code = 1; }
  }

  return exit_code;
}
