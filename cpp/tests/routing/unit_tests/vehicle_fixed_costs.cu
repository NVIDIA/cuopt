/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/solve.hpp>
#include <routing/utilities/check_constraints.hpp>
#include <utilities/copy_helpers.hpp>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

TEST(vehicle_fixed_costs, unlimited_fleet)
{
  int nlocations                 = 10;
  int norders                    = 10;
  int nvehicles                  = 20;
  std::vector<float> cost_matrix = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  std::vector<int> demands(norders, 1);
  demands[0] = 0;

  std::vector<int> capacities(nvehicles, 2);
  std::vector<float> fixed_vehicle_fixed_costs{16., 16., 16., 16., 16., 16., 16., 16., 16., 16.,
                                               1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix = cuopt::device_copy(cost_matrix, stream);

  auto v_capacities  = cuopt::device_copy(capacities, stream);
  auto v_demands     = cuopt::device_copy(demands, stream);
  auto v_fixed_costs = cuopt::device_copy(fixed_vehicle_fixed_costs, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());
  data_model.set_vehicle_fixed_costs(v_fixed_costs.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(5);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);
  constexpr const auto target = 19;  // 3 * 4 + 2 * 1 cost + vehicles costs 1 * 5
  ASSERT_FLOAT_EQ(routing_solution.get_total_objective(), target);
}

TEST(vehicle_fixed_costs, limited_fleet)
{
  int nlocations                 = 10;
  int norders                    = 10;
  int nvehicles                  = 16;
  std::vector<float> cost_matrix = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  std::vector<int> demands(norders, 1);
  demands[0] = 0;

  std::vector<int> capacities(nvehicles, 2);
  std::vector<float> fixed_vehicle_fixed_costs{
    16., 16., 16., 16., 16., 16., 16., 16., 16., 16., 16., 16., 16., 1., 1., 1.};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix = cuopt::device_copy(cost_matrix, stream);

  auto v_capacities  = cuopt::device_copy(capacities, stream);
  auto v_demands     = cuopt::device_copy(demands, stream);
  auto v_fixed_costs = cuopt::device_copy(fixed_vehicle_fixed_costs, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());
  data_model.set_vehicle_fixed_costs(v_fixed_costs.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(5);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);
  constexpr const auto target = 49;  // 3 * 4 + 2 * 1 cost + vehicles costs 1 * 3 + 2 * 16
  ASSERT_FLOAT_EQ(routing_solution.get_total_objective(), target);
}

TEST(vehicle_fixed_costs, with_objective)
{
  int nlocations                 = 5;
  int norders                    = 5;
  int nvehicles                  = 3;
  std::vector<float> cost_matrix = {0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
                                    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0};

  std::vector<int> demands(norders, 10);
  demands[0] = 0;

  std::vector<int> capacities = {50, 30, 30};
  std::vector<float> fixed_vehicle_fixed_costs{50., 10., 10.};
  std::vector<cuopt::routing::objective_t> objectives = {
    cuopt::routing::objective_t::COST, cuopt::routing::objective_t::VEHICLE_FIXED_COST};
  std::vector<float> objective_weights = {0.f, 1.f};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix = cuopt::device_copy(cost_matrix, stream);

  auto v_capacities        = cuopt::device_copy(capacities, stream);
  auto v_demands           = cuopt::device_copy(demands, stream);
  auto v_fixed_costs       = cuopt::device_copy(fixed_vehicle_fixed_costs, stream);
  auto v_objectives        = cuopt::device_copy(objectives, stream);
  auto v_objective_weights = cuopt::device_copy(objective_weights, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());
  data_model.set_vehicle_fixed_costs(v_fixed_costs.data());
  data_model.set_objective_function(
    v_objectives.data(), v_objective_weights.data(), v_objective_weights.size());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(7);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);
  constexpr const auto target = 20;
  ASSERT_FLOAT_EQ(routing_solution.get_total_objective(), target);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
