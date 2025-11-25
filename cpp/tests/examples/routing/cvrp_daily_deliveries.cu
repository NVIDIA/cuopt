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

bool basic_delivery_scenario()
{
  // 5 locations: depot (0) and 4 delivery points
  std::vector<float> cost_matrix = {
    0,  10, 15, 20, 25,  // depot to delivery points
    10, 0,  35, 25, 30,  // delivery point 1 to others
    15, 35, 0,  30, 20,  // delivery point 2 to others
    20, 25, 30, 0,  15,  // delivery point 3 to others
    25, 30, 20, 15, 0    // delivery point 4 to others
  };

  std::vector<int> order_locations         = {1, 2, 3, 4};  // Delivery points
  std::vector<int> vehicle_start_locations = {0, 0, 0};     // 3 vehicles starting from depot
  std::vector<int> vehicle_end_locations   = {0, 0, 0};     // All vehicles return to depot

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix             = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations         = cuopt::device_copy(order_locations, stream);
  auto v_vehicle_start_locations = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_vehicle_end_locations   = cuopt::device_copy(vehicle_end_locations, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 5, 3, 4);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_vehicle_locations(v_vehicle_start_locations.data(),
                                   v_vehicle_end_locations.data());

  auto routing_solution = cuopt::routing::solve(data_model);

  handle.sync_stream();
  if (routing_solution.get_status() != cuopt::routing::solution_status_t::SUCCESS) { return false; }

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);
  // Assuming the check_route function performs necessary checks, return true if no exceptions are
  // thrown
  return true;
}

bool capacity_constrained_deliveries()
{
  int nlocations = 6;  // depot + 5 delivery points
  int nvehicles  = 3;  // 3 delivery vehicles
  int norders    = 5;  // 5 delivery points

  // Cost matrix representing distances between locations
  std::vector<float> cost_matrix = {
    0,  10, 15, 20, 25, 30,  // depot to delivery points
    10, 0,  35, 25, 30, 20,  // delivery point 1 to others
    15, 35, 0,  30, 20, 25,  // delivery point 2 to others
    20, 25, 30, 0,  15, 30,  // delivery point 3 to others
    25, 30, 20, 15, 0,  35,  // delivery point 4 to others
    30, 20, 25, 30, 35, 0    // delivery point 5 to others
  };

  std::vector<int> order_locations         = {1, 2, 3, 4, 5};  // Delivery points
  std::vector<int> vehicle_start_locations = {0, 0, 0};        // All vehicles start from depot
  std::vector<int> vehicle_end_locations   = {0, 0, 0};        // All vehicles return to depot

  // Vehicle capacities (in units)
  std::vector<int> capacities = {100, 100, 100};

  // Delivery demands at each location (in units)
  std::vector<int> demands = {40, 30, 25, 35, 20};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_start_locations = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_end_locations   = cuopt::device_copy(vehicle_end_locations, stream);
  auto v_capacities      = cuopt::device_copy(capacities, stream);
  auto v_demands         = cuopt::device_copy(demands, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_vehicle_locations(v_start_locations.data(), v_end_locations.data());
  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());

  auto routing_solution = cuopt::routing::solve(data_model);

  handle.sync_stream();
  if (routing_solution.get_status() != cuopt::routing::solution_status_t::SUCCESS) { return false; }

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);
  // Assuming the check_route function performs necessary checks, return true if no exceptions are
  // thrown
  return true;
}

bool minimum_vehicles_constraint()
{
  int nlocations = 7;  // depot + 6 delivery points
  int nvehicles  = 4;  // 4 available vehicles
  int norders    = 6;  // 6 delivery points

  // Cost matrix representing distances between locations
  std::vector<float> cost_matrix = {
    0,  10, 15, 20, 25, 30, 35,  // depot to delivery points
    10, 0,  35, 25, 30, 20, 25,  // delivery point 1 to others
    15, 35, 0,  30, 20, 25, 30,  // delivery point 2 to others
    20, 25, 30, 0,  15, 30, 35,  // delivery point 3 to others
    25, 30, 20, 15, 0,  35, 40,  // delivery point 4 to others
    30, 20, 25, 30, 35, 0,  45,  // delivery point 5 to others
    35, 25, 30, 35, 40, 45, 0    // delivery point 6 to others
  };

  std::vector<int> order_locations         = {1, 2, 3, 4, 5, 6};  // Delivery points
  std::vector<int> vehicle_start_locations = {0, 0, 0, 0};        // All vehicles start from depot
  std::vector<int> vehicle_end_locations   = {0, 0, 0, 0};        // All vehicles return to depot

  // Vehicle capacities (in units)
  std::vector<int> capacities = {100, 100, 100, 100};

  // Delivery demands at each location (in units)
  std::vector<int> demands = {40, 30, 25, 35, 20, 30};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_start_locations = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_end_locations   = cuopt::device_copy(vehicle_end_locations, stream);
  auto v_capacities      = cuopt::device_copy(capacities, stream);
  auto v_demands         = cuopt::device_copy(demands, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_vehicle_locations(v_start_locations.data(), v_end_locations.data());
  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());

  // Set minimum number of vehicles to 3
  data_model.set_min_vehicles(3);

  auto routing_solution = cuopt::routing::solve(data_model);

  handle.sync_stream();
  if (routing_solution.get_status() != cuopt::routing::solution_status_t::SUCCESS) { return false; }

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);
  // Assuming the check_route function performs necessary checks, return true if no exceptions are
  // thrown
  return true;
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt

int main()
{
  std::vector<std::string> example_names = {
    "basic_delivery_scenario", "capacity_constrained_deliveries", "minimum_vehicles_constraint"};

  std::vector<std::function<bool()>> example_functions = {
    cuopt::routing::test::basic_delivery_scenario,
    cuopt::routing::test::capacity_constrained_deliveries,
    cuopt::routing::test::minimum_vehicles_constraint};

  int exit_code = 0;
  for (size_t i = 0; i < example_names.size(); ++i) {
    int status = example_functions[i]();
    std::cout << "Example: " << example_names[i]
              << " - Status: " << (status ? "Success" : "Failure") << std::endl;
    if (status) { exit_code = 1; }
  }

  return exit_code;
}
