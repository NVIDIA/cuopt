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

bool basic_service_team_scenario()
{
  // 5 locations: depot (0) and 4 service points
  int nlocations = 5;
  int nvehicles  = 3;  // 3 service teams
  int norders    = 4;  // 4 service requests

  // Cost matrix representing distances between locations
  std::vector<float> cost_matrix = {
    0,  10, 15, 20, 25,  // depot to service points
    10, 0,  35, 25, 30,  // service point 1 to others
    15, 35, 0,  30, 20,  // service point 2 to others
    20, 25, 30, 0,  15,  // service point 3 to others
    25, 30, 20, 15, 0    // service point 4 to others
  };

  // Service request locations
  std::vector<int> order_locations = {1, 2, 3, 4};

  // Time windows for service requests (earliest and latest times)
  std::vector<int> order_earliest = {0, 0, 0, 0};
  std::vector<int> order_latest   = {100, 100, 100, 100};

  // Service times at each location
  std::vector<int> order_service_times = {30, 45, 60, 30};

  // Break constraints for service teams
  std::vector<int> break_earliest  = {30, 30, 30};  // Earliest break times
  std::vector<int> break_latest    = {70, 70, 70};  // Latest break times
  std::vector<int> break_duration  = {15, 15, 15};  // Break duration in minutes
  std::vector<int> break_locations = {0, 0, 0};     // Breaks must be taken at depot

  // Vehicle start and end locations (all at depot)
  std::vector<int> vehicle_start_locations = {0, 0, 0};
  std::vector<int> vehicle_end_locations   = {0, 0, 0};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  // Copy data to device
  auto v_cost_matrix         = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_order_earliest      = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest        = cuopt::device_copy(order_latest, stream);
  auto v_order_service_times = cuopt::device_copy(order_service_times, stream);
  auto v_break_earliest      = cuopt::device_copy(break_earliest, stream);
  auto v_break_latest        = cuopt::device_copy(break_latest, stream);
  auto v_break_duration      = cuopt::device_copy(break_duration, stream);
  auto v_break_locations     = cuopt::device_copy(break_locations, stream);
  auto v_start_locations     = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_end_locations       = cuopt::device_copy(vehicle_end_locations, stream);

  // Create and configure the data model
  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);

  // Add cost matrix
  data_model.add_cost_matrix(v_cost_matrix.data());

  // Set order locations and time windows
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_order_service_times(v_order_service_times.data());

  // Set vehicle locations
  data_model.set_vehicle_locations(v_start_locations.data(), v_end_locations.data());

  // Add break constraints
  data_model.add_break_dimension(
    v_break_earliest.data(), v_break_latest.data(), v_break_duration.data());
  data_model.set_break_locations(v_break_locations.data(), v_break_locations.size());

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

  // Assuming the check_route function performs necessary checks, return true if no exceptions are
  // thrown
  return true;
}

bool complex_service_team_scenario()
{
  // 8 locations: depot (0) and 7 service points
  int nlocations = 8;
  int nvehicles  = 4;  // 4 service teams
  int norders    = 6;  // 6 service requests

  // Cost matrix representing distances between locations
  std::vector<float> cost_matrix = {
    0,  10, 15, 20, 25, 30, 35, 40,  // depot to service points
    10, 0,  35, 25, 30, 20, 25, 30,  // service point 1 to others
    15, 35, 0,  30, 20, 25, 30, 35,  // service point 2 to others
    20, 25, 30, 0,  15, 30, 35, 40,  // service point 3 to others
    25, 30, 20, 15, 0,  35, 40, 45,  // service point 4 to others
    30, 20, 25, 30, 35, 0,  45, 50,  // service point 5 to others
    35, 25, 30, 35, 40, 45, 0,  55,  // service point 6 to others
    40, 30, 35, 40, 45, 50, 55, 0    // service point 7 to others
  };

  // Service request locations
  std::vector<int> order_locations = {1, 2, 3, 4, 5, 6};

  // Time windows for service requests (earliest and latest times)
  std::vector<int> order_earliest = {0, 0, 0, 0, 0, 0};
  std::vector<int> order_latest   = {200, 200, 200, 200, 200, 200};

  // Service times at each location
  std::vector<int> order_service_times = {45, 60, 30, 75, 45, 60};

  // Break constraints for service teams
  std::vector<int> break_earliest  = {60, 60, 60, 60};      // Earliest break times
  std::vector<int> break_latest    = {140, 140, 140, 140};  // Latest break times
  std::vector<int> break_duration  = {30, 30, 30, 30};      // Break duration in minutes
  std::vector<int> break_locations = {0, 0, 0, 0};          // Breaks must be taken at depot

  // Vehicle start and end locations (all at depot)
  std::vector<int> vehicle_start_locations = {0, 0, 0, 0};
  std::vector<int> vehicle_end_locations   = {0, 0, 0, 0};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  // Copy data to device
  auto v_cost_matrix         = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_order_earliest      = cuopt::device_copy(order_earliest, stream);
  auto v_order_latest        = cuopt::device_copy(order_latest, stream);
  auto v_order_service_times = cuopt::device_copy(order_service_times, stream);
  auto v_break_earliest      = cuopt::device_copy(break_earliest, stream);
  auto v_break_latest        = cuopt::device_copy(break_latest, stream);
  auto v_break_duration      = cuopt::device_copy(break_duration, stream);
  auto v_break_locations     = cuopt::device_copy(break_locations, stream);
  auto v_start_locations     = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_end_locations       = cuopt::device_copy(vehicle_end_locations, stream);

  // Create and configure the data model
  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);

  // Add cost matrix
  data_model.add_cost_matrix(v_cost_matrix.data());

  // Set order locations and time windows
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_time_windows(v_order_earliest.data(), v_order_latest.data());
  data_model.set_order_service_times(v_order_service_times.data());

  // Set vehicle locations
  data_model.set_vehicle_locations(v_start_locations.data(), v_end_locations.data());

  // Add break constraints
  data_model.add_break_dimension(
    v_break_earliest.data(), v_break_latest.data(), v_break_duration.data());
  data_model.set_break_locations(v_break_locations.data(), v_break_locations.size());

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

  // Assuming the check_route function performs necessary checks, return true if no exceptions are
  // thrown
  return true;
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt

int main()
{
  std::vector<std::string> example_names = {"basic_service_team_scenario",
                                            "complex_service_team_scenario"};

  std::vector<std::function<bool()>> example_functions = {
    cuopt::routing::test::basic_service_team_scenario,
    cuopt::routing::test::complex_service_team_scenario};

  int exit_code = 0;
  for (size_t i = 0; i < example_names.size(); ++i) {
    int status = example_functions[i]();
    std::cout << "Example: " << example_names[i]
              << " - Status: " << (status ? "Success" : "Failure") << std::endl;
    if (!status) { exit_code = 1; }
  }

  return exit_code;
}
