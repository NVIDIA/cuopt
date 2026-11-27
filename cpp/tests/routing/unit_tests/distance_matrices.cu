/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/cpu_routing_problem.hpp>
#include <cuopt/routing/solve.hpp>
#include <utilities/copy_helpers.hpp>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuopt::routing::test {

TEST(distance_matrices, maximum_distance_selects_vehicle_independently_of_cost_and_time)
{
  cpu_routing_problem_t problem;
  problem.num_locations         = 2;
  problem.fleet_size            = 2;
  problem.num_orders            = 1;
  problem.order_locations       = {1};
  problem.vehicle_types         = {7, 3};
  problem.cost_matrices         = {{7, {0, 1, 2, 0}}, {3, {0, 9, 8, 0}}};
  problem.distance_matrices     = {{3, {0, 3, 4, 0}}, {7, {0, 10, 20, 0}}};
  problem.transit_time_matrices = {{7, {0, 4, 6, 0}}, {3, {0, 5, 7, 0}}};
  raft::handle_t handle;
  solver_settings_t<int, float> settings;
  settings.set_time_limit(1);
  const auto check = [&](std::vector<float> const& max_costs,
                         std::vector<float> const& max_distances,
                         int expected_vehicle) {
    SCOPED_TRACE(::testing::Message()
                 << "max costs: " << ::testing::PrintToString(max_costs)
                 << ", max distances: " << ::testing::PrintToString(max_distances));
    problem.vehicle_max_costs     = max_costs;
    problem.vehicle_max_distances = max_distances;
    auto [model, storage]         = problem.to_device(&handle);
    auto solution                 = solve(model, settings);
    ASSERT_EQ(solution.get_status(), solution_status_t::SUCCESS);
    host_assignment_t<int> result(solution);
    EXPECT_EQ(result.truck_id, (std::vector<int>(3, expected_vehicle)));
    EXPECT_DOUBLE_EQ(solution.get_total_objective(), expected_vehicle == 0 ? 3 : 17);
    EXPECT_EQ(
      result.stamp,
      (expected_vehicle == 0 ? std::vector<double>{0, 4, 10} : std::vector<double>{0, 5, 12}));
  };
  check({3, 17}, {}, 0);
  check({3, 17}, {29, 7}, 1);
  check({100, 100}, {29, 7}, 1);  // Vehicle 0 is within budget but travels 30.
  check({2, 17}, {100, 7}, 1);    // Vehicle 0 is within range but costs 3.
}

TEST(distance_matrices, maximum_distance_counts_only_travelled_legs)
{
  cpu_routing_problem_t problem;
  problem.num_locations     = 2;
  problem.fleet_size        = 1;
  problem.num_orders        = 1;
  problem.cost_matrices     = {{0, {0, 2, 3, 0}}};
  problem.distance_matrices = {{0, {0, 7, 13, 0}}};
  problem.order_locations   = {1};
  raft::handle_t handle;
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.5);
  for (bool skip_first : {false, true}) {
    for (bool drop_return : {false, true}) {
      SCOPED_TRACE(::testing::Message() << skip_first << ", " << drop_return);
      problem.skip_first_trips      = {static_cast<uint8_t>(skip_first)};
      problem.drop_return_trips     = {static_cast<uint8_t>(drop_return)};
      float distance                = (skip_first ? 0 : 7) + (drop_return ? 0 : 13);
      problem.vehicle_max_distances = {distance};
      auto [model, storage]         = problem.to_device(&handle);
      auto solution                 = solve(model, settings);
      ASSERT_EQ(solution.get_status(), solution_status_t::SUCCESS);
      EXPECT_DOUBLE_EQ(solution.get_total_objective(),
                       (skip_first ? 0 : 2) + (drop_return ? 0 : 3));
      if (distance > 0) {
        problem.vehicle_max_distances = {distance - 1};
        auto [limited, buffers]       = problem.to_device(&handle);
        EXPECT_EQ(solve(limited, settings).get_status(), solution_status_t::INFEASIBLE);
      }
    }
  }
}

TEST(distance_matrices, maximum_distance_can_require_a_more_expensive_tour)
{
  cpu_routing_problem_t problem;
  problem.num_locations         = 3;
  problem.num_orders            = 2;
  problem.fleet_size            = 1;
  problem.order_locations       = {1, 2};
  problem.cost_matrices         = {{0, {0, 1, 5, 5, 0, 1, 1, 5, 0}}};
  problem.distance_matrices     = {{0, {0, 10, 2, 2, 0, 10, 10, 2, 0}}};
  problem.vehicle_max_distances = {6};

  raft::handle_t handle;
  auto [model, storage] = problem.to_device(&handle);
  solver_settings_t<int, float> settings;
  settings.set_time_limit(1);
  auto solution = solve(model, settings);
  ASSERT_EQ(solution.get_status(), solution_status_t::SUCCESS);
  host_assignment_t<int> result(solution);
  EXPECT_EQ(result.locations, (std::vector<int>{0, 2, 1, 0}));
  EXPECT_DOUBLE_EQ(solution.get_total_objective(), 15);
}

TEST(distance_matrices, maximum_distance_validates_bounds_and_requires_distance_matrix)
{
  raft::handle_t handle;
  cpu_routing_problem_t problem;
  problem.num_locations         = 2;
  problem.fleet_size            = 1;
  problem.num_orders            = 1;
  problem.order_locations       = {1};
  problem.cost_matrices         = {{0, {0, 1, 2, 0}}};
  problem.vehicle_max_distances = {1, 2};
  EXPECT_THROW(problem.to_device(&handle), std::invalid_argument);
  problem.vehicle_max_distances = {0};
  auto [model, storage]         = problem.to_device(&handle);
  EXPECT_THROW(model.set_vehicle_max_distances(nullptr), cuopt::logic_error);
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.1);
  auto missing = solve(model, settings);
  ASSERT_EQ(missing.get_status(), solution_status_t::ERROR);
  EXPECT_NE(std::string(missing.get_error_status().what()).find("require a distance matrix"),
            std::string::npos);
  model.add_distance_matrix(model.get_cost_matrix());
  for (float invalid :
       {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    problem.vehicle_max_distances = {invalid};
    EXPECT_THROW(problem.to_device(&handle), std::invalid_argument);
    auto limits = cuopt::device_copy(std::vector<float>{invalid}, handle.get_stream());
    model.set_vehicle_max_distances(limits.data());
    auto solution = solve(model, settings);
    ASSERT_EQ(solution.get_status(), solution_status_t::ERROR);
    EXPECT_NE(std::string(solution.get_error_status().what()).find("finite and non-negative"),
              std::string::npos);
  }
}

TEST(distance_matrices, limits_belong_to_vehicles_sharing_a_matrix)
{
  cpu_routing_problem_t problem;
  problem.num_locations     = 3;
  problem.fleet_size        = 2;
  problem.num_orders        = 2;
  problem.order_locations   = {1, 2};
  problem.pickup_indices    = {0};
  problem.delivery_indices  = {1};
  problem.cost_matrices     = {{0, {0, 2, 6, 5, 0, 3, 4, 7, 0}}};
  problem.distance_matrices = {{0, {0, 3, 8, 8, 0, 5, 7, 8, 0}}};
  raft::handle_t handle;
  solver_settings_t<int, float> settings;
  settings.set_time_limit(1);
  for (int eligible : {0, 1}) {
    SCOPED_TRACE(eligible);
    problem.vehicle_max_distances           = {14, 14};
    problem.vehicle_max_distances[eligible] = 15;
    auto [model, storage]                   = problem.to_device(&handle);
    EXPECT_EQ(cuopt::host_copy(model.get_vehicle_max_distances(), handle.get_stream()),
              problem.vehicle_max_distances);
    auto solution = solve(model, settings);
    ASSERT_EQ(solution.get_status(), solution_status_t::SUCCESS);
    host_assignment_t<int> result(solution);
    EXPECT_EQ(result.locations, (std::vector<int>{0, 1, 2, 0}));
    EXPECT_EQ(result.truck_id, (std::vector<int>(4, eligible)));
    EXPECT_DOUBLE_EQ(solution.get_total_objective(), 9);
  }
}

TEST(distance_matrices, unreachable_leg_exceeds_even_the_largest_finite_limit)
{
  cpu_routing_problem_t problem;
  problem.num_locations         = 2;
  problem.fleet_size            = 1;
  problem.num_orders            = 1;
  problem.order_locations       = {1};
  problem.cost_matrices         = {{0, {0, 2, 3, 0}}};
  problem.distance_matrices     = {{0, {0, 0, std::numeric_limits<float>::infinity(), 0}}};
  problem.vehicle_max_distances = {std::numeric_limits<float>::max()};
  raft::handle_t handle;
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.5);
  for (bool drop_return : {false, true}) {
    SCOPED_TRACE(drop_return);
    problem.drop_return_trips = {static_cast<uint8_t>(drop_return)};
    auto [model, storage]     = problem.to_device(&handle);
    auto solution             = solve(model, settings);
    EXPECT_EQ(solution.get_status(),
              drop_return ? solution_status_t::SUCCESS : solution_status_t::INFEASIBLE)
      << solution.get_error_status().what();
    if (drop_return) { EXPECT_DOUBLE_EQ(solution.get_total_objective(), 2); }
  }
}

TEST(distance_matrices, uses_only_fleet_types_from_a_full_matrix_catalog)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();
  auto cost   = cuopt::device_copy(std::vector<float>{0, 2, 3, 0}, stream);
  auto dist   = cuopt::device_copy(std::vector<float>{0, 4, 6, 0}, stream);
  auto type   = cuopt::device_copy(std::vector<uint8_t>{255}, stream);
  auto limit  = cuopt::device_copy(std::vector<float>{10}, stream);
  auto order  = cuopt::device_copy(std::vector<int>{1}, stream);
  data_model_view_t<int, float> model(&handle, 2, 1, 1);
  for (int id = 0; id <= std::numeric_limits<uint8_t>::max(); ++id) {
    model.add_cost_matrix(cost.data(), id);
  }
  model.add_distance_matrix(dist.data(), 255);
  model.set_vehicle_types(type.data());
  model.set_order_locations(order.data());
  EXPECT_TRUE(model.get_vehicle_max_distances().empty());
  model.set_vehicle_max_distances(limit.data());
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.5);
  auto solution = solve(model, settings);
  ASSERT_EQ(solution.get_status(), solution_status_t::SUCCESS);
  EXPECT_DOUBLE_EQ(solution.get_total_objective(), 5);
}

TEST(distance_matrices, cpu_problem_rejects_incomplete_vehicle_types)
{
  raft::handle_t handle;
  cpu_routing_problem_t problem;
  problem.num_locations         = 2;
  problem.fleet_size            = 2;
  problem.cost_matrices         = {{0, {0, 1, 1, 0}}};
  problem.distance_matrices     = {{0, {0, 2, 2, 0}}};
  problem.vehicle_max_distances = {4, 5};
  problem.vehicle_types         = {0};
  EXPECT_THROW(problem.to_device(&handle), std::invalid_argument);
}

TEST(distance_matrices, optional_distance_preserves_cost_and_arrival_times)
{
  raft::handle_t handle;
  auto stream   = handle.get_stream();
  auto cost     = cuopt::device_copy(std::vector<float>{0.f, 3.f, 5.f, 0.f}, stream);
  auto distance = cuopt::device_copy(std::vector<float>{0.f, 100.f, 200.f, 0.f}, stream);
  auto time     = cuopt::device_copy(std::vector<float>{0.f, 11.f, 17.f, 0.f}, stream);
  auto order    = cuopt::device_copy(std::vector<int>{1}, stream);
  auto earliest = cuopt::device_copy(std::vector<int>{0}, stream);
  auto latest   = cuopt::device_copy(std::vector<int>{1000}, stream);
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.1);

  for (bool with_time : {false, true}) {
    for (bool with_distance : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "with_time=" << with_time << ", with_distance=" << with_distance);
      data_model_view_t<int, float> model(&handle, 2, 1, 1);
      model.add_cost_matrix(cost.data());
      model.set_order_locations(order.data());
      model.set_order_time_windows(earliest.data(), latest.data());
      if (with_time) { model.add_transit_time_matrix(time.data()); }
      if (with_distance) { model.add_distance_matrix(distance.data()); }

      auto solution = solve(model, settings);
      ASSERT_EQ(solution.get_status(), solution_status_t::SUCCESS);
      EXPECT_DOUBLE_EQ(solution.get_total_objective(), 8.);
      host_assignment_t<int> result(solution);
      EXPECT_EQ(result.locations, (std::vector<int>{0, 1, 0}));
      auto expected_arrivals =
        with_time ? std::vector<double>{0., 11., 28.} : std::vector<double>{0., 3., 8.};
      EXPECT_EQ(result.stamp, expected_arrivals);
    }
  }
}

TEST(distance_matrices, cpu_problem_keeps_three_metrics_separate)
{
  cpu_routing_problem_t problem;
  problem.num_locations         = 2;
  problem.fleet_size            = 1;
  problem.num_orders            = 1;
  problem.cost_matrices         = {{0, {0.f, 3.f, 5.f, 0.f}}};
  problem.distance_matrices     = {{0, {0.f, 100.f, 200.f, 0.f}}};
  problem.transit_time_matrices = {{0, {0.f, 11.f, 17.f, 0.f}}};
  problem.order_locations       = {1};
  raft::handle_t handle;
  auto [model, storage] = problem.to_device(&handle);
  ASSERT_NE(model.get_distance_matrix(), nullptr);
  auto distance = cuopt::host_copy(model.get_distance_matrix(), 4, handle.get_stream());
  EXPECT_EQ(distance, (std::vector<float>{0.f, 100.f, 200.f, 0.f}));
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.1);
  auto solution = solve(model, settings);
  ASSERT_EQ(solution.get_status(), solution_status_t::SUCCESS);
  EXPECT_DOUBLE_EQ(solution.get_total_objective(), 8.);
  host_assignment_t<int> result(solution);
  EXPECT_EQ(result.locations, (std::vector<int>{0, 1, 0}));
  EXPECT_EQ(result.stamp, (std::vector<double>{0., 11., 28.}));
}

TEST(distance_matrices, cpu_problem_rejects_invalid_distance_data)
{
  cpu_routing_problem_t problem;
  problem.num_locations = 2;
  problem.fleet_size    = 1;
  problem.num_orders    = 1;
  problem.cost_matrices = {{0, {0.f, 3.f, 5.f, 0.f}}};
  raft::handle_t handle;
  for (auto const& values :
       {std::vector<float>{},
        std::vector<float>{0.f, 1.f},
        std::vector<float>{0.f, -1.f, 1.f, 0.f},
        std::vector<float>{0.f, std::numeric_limits<float>::quiet_NaN(), 1.f, 0.f},
        std::vector<float>{0.f, -std::numeric_limits<float>::infinity(), 1.f, 0.f}}) {
    problem.distance_matrices = {{0, values}};
    EXPECT_THROW(problem.to_device(&handle), std::invalid_argument);
  }
}

TEST(distance_matrices, distance_matrix_does_not_satisfy_maximum_time_requirement)
{
  cpu_routing_problem_t problem;
  problem.num_locations     = 2;
  problem.fleet_size        = 1;
  problem.num_orders        = 1;
  problem.cost_matrices     = {{0, {0.f, 3.f, 5.f, 0.f}}};
  problem.distance_matrices = {{0, {0.f, 100.f, 200.f, 0.f}}};
  problem.vehicle_max_times = {50.f};
  problem.order_locations   = {1};
  raft::handle_t handle;
  auto [model, storage] = problem.to_device(&handle);
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.1);
  auto solution = solve(model, settings);
  EXPECT_EQ(solution.get_status(), solution_status_t::ERROR);
  EXPECT_NE(std::string(solution.get_error_status().what()).find("Time matrix should be set"),
            std::string::npos);
}

TEST(distance_matrices, cpu_problem_requires_matching_sizes_for_all_matrices)
{
  cpu_routing_problem_t problem;
  problem.num_locations     = 2;
  problem.fleet_size        = 1;
  problem.num_orders        = 1;
  problem.distance_matrices = {{0, {0.f, 1.f, 1.f, 0.f}}};
  problem.cost_matrices     = {{0, {0.f}}};
  raft::handle_t handle;
  EXPECT_THROW(problem.to_device(&handle), std::invalid_argument);
  problem.cost_matrices         = {{0, {0.f, 3.f, 5.f, 0.f}}};
  problem.transit_time_matrices = {{0, {0.f}}};
  EXPECT_THROW(problem.to_device(&handle), std::invalid_argument);
}

TEST(distance_matrices, time_constraints_preserve_vehicle_type_mapping_with_optional_distance)
{
  raft::handle_t handle;
  solver_settings_t<int, float> settings;
  settings.set_time_limit(1);
  for (bool with_distance : {false, true}) {
    for (bool with_time_for_first_vehicle : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "with_distance=" << with_distance
                   << ", with_time_for_first_vehicle=" << with_time_for_first_vehicle);
      cpu_routing_problem_t problem;
      problem.num_locations         = 2;
      problem.fleet_size            = 2;
      problem.num_orders            = 1;
      problem.cost_matrices         = {{3, {0.f, 3.f, 5.f, 0.f}}, {1, {0.f, 7.f, 9.f, 0.f}}};
      problem.transit_time_matrices = {{1, {0.f, 11.f, 17.f, 0.f}}};
      if (with_time_for_first_vehicle) {
        problem.transit_time_matrices.push_back({3, {0.f, 13.f, 31.f, 0.f}});
      }
      if (with_distance) {
        problem.distance_matrices = {{3, {0.f, 1.e9f, 2.e9f, 0.f}}, {1, {0.f, 3.e9f, 4.e9f, 0.f}}};
      }
      problem.vehicle_types     = {3, 1};
      problem.vehicle_max_times = {30.f, with_time_for_first_vehicle ? 30.f : 20.f};
      problem.order_locations   = {1};
      auto [model, storage]     = problem.to_device(&handle);
      auto solution             = solve(model, settings);
      ASSERT_EQ(solution.get_status(), solution_status_t::SUCCESS);
      EXPECT_DOUBLE_EQ(solution.get_total_objective(), with_time_for_first_vehicle ? 16. : 8.);
      host_assignment_t<int> result(solution);
      EXPECT_EQ(result.locations, (std::vector<int>{0, 1, 0}));
      auto vehicle = with_time_for_first_vehicle ? 1 : 0;
      EXPECT_EQ(result.truck_id, (std::vector<int>{vehicle, vehicle, vehicle}));
      auto arrivals = with_time_for_first_vehicle ? std::vector<double>{0., 11., 28.}
                                                  : std::vector<double>{0., 3., 8.};
      EXPECT_EQ(result.stamp, arrivals);
    }
  }
}

TEST(distance_matrices, unused_vehicle_type_time_does_not_satisfy_maximum_time_requirement)
{
  raft::handle_t handle;
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.1);
  for (bool with_distance : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "with_distance=" << with_distance);
    cpu_routing_problem_t problem;
    problem.num_locations         = 2;
    problem.fleet_size            = 1;
    problem.num_orders            = 1;
    problem.cost_matrices         = {{0, {0.f, 3.f, 5.f, 0.f}}, {3, {0.f, 7.f, 9.f, 0.f}}};
    problem.transit_time_matrices = {{3, {0.f, 11.f, 17.f, 0.f}}};
    if (with_distance) { problem.distance_matrices = {{0, {0.f, 100.f, 200.f, 0.f}}}; }
    problem.vehicle_types     = {0};
    problem.vehicle_max_times = {50.f};
    problem.order_locations   = {1};
    auto [model, storage]     = problem.to_device(&handle);
    auto solution             = solve(model, settings);
    EXPECT_EQ(solution.get_status(), solution_status_t::ERROR);
    EXPECT_NE(std::string(solution.get_error_status().what()).find("Time matrix should be set"),
              std::string::npos);
  }
}

TEST(distance_matrices, registration_is_optional_and_null_input_is_rejected)
{
  raft::handle_t handle;
  data_model_view_t<int, float> model(&handle, 2, 1, 1);
  EXPECT_EQ(model.get_distance_matrix(), nullptr);
  EXPECT_TRUE(model.get_distance_matrices().empty());
  EXPECT_THROW(model.add_distance_matrix(nullptr), cuopt::logic_error);
  auto distance = cuopt::device_copy(std::vector<float>{0.f, 5.f, 7.f, 0.f}, handle.get_stream());
  model.add_distance_matrix(distance.data(), 3);
  EXPECT_EQ(model.get_distance_matrix(3), distance.data());
  EXPECT_EQ(model.get_distance_matrices().size(), 1u);
  EXPECT_EQ(model.get_distance_matrix(0), nullptr);
}

TEST(distance_matrices, device_input_is_validated_before_solving)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();
  auto cost   = cuopt::device_copy(std::vector<float>{0.f, 3.f, 5.f, 0.f}, stream);
  auto order  = cuopt::device_copy(std::vector<int>{1}, stream);
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.1);
  for (float invalid :
       {-1.f, std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::infinity()}) {
    auto distance = cuopt::device_copy(std::vector<float>{0.f, invalid, 1.f, 0.f}, stream);
    data_model_view_t<int, float> model(&handle, 2, 1, 1);
    model.add_cost_matrix(cost.data());
    model.add_distance_matrix(distance.data());
    model.set_order_locations(order.data());
    auto solution = solve(model, settings);
    EXPECT_EQ(solution.get_status(), solution_status_t::ERROR);
    EXPECT_NE(std::string(solution.get_error_status().what()).find("non-negative and not NaN"),
              std::string::npos);
  }
}

TEST(distance_matrices, distance_vehicle_types_require_corresponding_cost_matrices)
{
  cpu_routing_problem_t problem;
  problem.num_locations     = 2;
  problem.fleet_size        = 1;
  problem.num_orders        = 1;
  problem.cost_matrices     = {{0, {0.f, 3.f, 5.f, 0.f}}};
  problem.distance_matrices = {{1, {0.f, 100.f, 200.f, 0.f}}};
  problem.order_locations   = {1};
  raft::handle_t handle;
  auto [model, storage] = problem.to_device(&handle);
  solver_settings_t<int, float> settings;
  auto solution = solve(model, settings);
  EXPECT_EQ(solution.get_status(), solution_status_t::ERROR);
}

TEST(distance_matrices, auxiliary_distance_input_is_preserved_after_solving)
{
  cpu_routing_problem_t problem;
  problem.num_locations         = 2;
  problem.fleet_size            = 1;
  problem.num_orders            = 1;
  problem.cost_matrices         = {{0, {0.f, 3.f, 5.f, 0.f}}};
  problem.distance_matrices     = {{0, {0.f, std::numeric_limits<float>::infinity(), 1.e30f, 0.f}}};
  problem.transit_time_matrices = {{0, {0.f, 11.f, 17.f, 0.f}}};
  problem.order_locations       = {1};
  raft::handle_t handle;
  auto [model, storage] = problem.to_device(&handle);
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.1);
  auto solution = solve(model, settings);
  ASSERT_EQ(solution.get_status(), solution_status_t::SUCCESS);
  EXPECT_DOUBLE_EQ(solution.get_total_objective(), 8.);
  host_assignment_t<int> result(solution);
  EXPECT_EQ(result.locations, (std::vector<int>{0, 1, 0}));
  EXPECT_EQ(result.stamp, (std::vector<double>{0., 11., 28.}));

  ASSERT_NE(model.get_distance_matrix(), nullptr);
  auto distance = cuopt::host_copy(model.get_distance_matrix(), 4, handle.get_stream());
  EXPECT_EQ(distance, problem.distance_matrices.front().matrix);
  ASSERT_NE(model.get_cost_matrix(), nullptr);
  auto cost = cuopt::host_copy(model.get_cost_matrix(), 4, handle.get_stream());
  EXPECT_EQ(cost, problem.cost_matrices.front().matrix);
  ASSERT_NE(model.get_transit_time_matrix(), nullptr);
  auto time = cuopt::host_copy(model.get_transit_time_matrix(), 4, handle.get_stream());
  EXPECT_EQ(time, problem.transit_time_matrices.front().matrix);
}

TEST(distance_matrices, omitted_distance_matrix_preserves_cost_only_model)
{
  cpu_routing_problem_t problem;
  problem.num_locations   = 2;
  problem.fleet_size      = 1;
  problem.num_orders      = 1;
  problem.cost_matrices   = {{0, {0.f, 3.f, 5.f, 0.f}}};
  problem.order_locations = {1};
  raft::handle_t handle;
  auto [model, storage] = problem.to_device(&handle);
  EXPECT_TRUE(model.get_distance_matrices().empty());
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.1);
  auto solution = solve(model, settings);
  EXPECT_EQ(solution.get_status(), solution_status_t::SUCCESS);
  EXPECT_DOUBLE_EQ(solution.get_total_objective(), 8.);
}

TEST(distance_matrices, partial_distance_coverage_of_fleet_is_rejected)
{
  cpu_routing_problem_t problem;
  problem.num_locations     = 2;
  problem.fleet_size        = 2;
  problem.num_orders        = 1;
  problem.cost_matrices     = {{1, {0.f, 3.f, 5.f, 0.f}}, {3, {0.f, 7.f, 9.f, 0.f}}};
  problem.distance_matrices = {{1, {0.f, 100.f, 200.f, 0.f}}};
  problem.vehicle_types     = {1, 3};
  problem.order_locations   = {1};
  raft::handle_t handle;
  auto [model, storage] = problem.to_device(&handle);
  solver_settings_t<int, float> settings;
  auto solution = solve(model, settings);
  EXPECT_EQ(solution.get_status(), solution_status_t::ERROR);
  EXPECT_NE(std::string(solution.get_error_status().what()).find("All vehicle distance matrices"),
            std::string::npos);
}

}  // namespace cuopt::routing::test
