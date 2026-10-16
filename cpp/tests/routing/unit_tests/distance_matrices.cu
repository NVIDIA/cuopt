/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/cpu_routing_problem.hpp>
#include <cuopt/routing/solve.hpp>
#include <routing/utilities/md_utils.hpp>
#include <routing/vehicle_info.hpp>
#include <utilities/copy_helpers.hpp>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuopt::routing::test {

TEST(distance_matrices, host_builders_preserve_legacy_time_slot)
{
  auto single = detail::create_host_mdarray<float>(2, 1, 1);
  EXPECT_EQ(single.get_time_matrix(0), single.get_cost_matrix(0));
  auto pair   = detail::create_host_mdarray<float>(2, 1, 2);
  pair.buffer = {0.f, 3.f, 5.f, 0.f, 0.f, 11.f, 17.f, 0.f};
  EXPECT_FLOAT_EQ(pair.get_time_matrix(0)[1], 11.f);
  EXPECT_FLOAT_EQ(pair.view().get_time_matrix(0)[1], 11.f);
}

TEST(distance_matrices, two_slot_distance_layout_does_not_report_transit_time)
{
  auto matrices                  = detail::create_host_mdarray<float>(2, 1, 2);
  matrices.distance_matrix_index = 1;
  matrices.time_matrix_index     = 0;
  matrices.buffer                = {0.f, 3.f, 5.f, 0.f, 0.f, 100.f, 200.f, 0.f};
  detail::VehicleInfo<float, false> vehicle;
  vehicle.matrices = matrices.view();
  EXPECT_FALSE(vehicle.has_time_matrix());
  EXPECT_FLOAT_EQ(matrices.view().get_cost_matrix(0)[1], 3.f);
  EXPECT_FLOAT_EQ(matrices.view().get_distance_matrix(0)[1], 100.f);
  EXPECT_FLOAT_EQ(matrices.view().get_time_matrix(0)[1], 3.f);
}

TEST(distance_matrices, cpu_problem_and_device_layout_keep_three_metrics_separate)
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
  auto matrices         = detail::create_device_mdarray<float>(2, 1, 3, handle.get_stream());
  detail::fill_mdarray_from_data_model(matrices, model);
  EXPECT_EQ(matrices.distance_matrix_index, 1);
  EXPECT_EQ(matrices.time_matrix_index, 2);
  auto copied =
    cuopt::host_copy(matrices.buffer.data(), matrices.buffer.size(), handle.get_stream());
  EXPECT_EQ(copied,
            (std::vector<float>{0.f, 3.f, 5.f, 0.f, 0.f, 100.f, 200.f, 0.f, 0.f, 11.f, 17.f, 0.f}));
  solver_settings_t<int, float> settings;
  settings.set_time_limit(0.1);
  auto solution = solve(model, settings);
  EXPECT_EQ(solution.get_status(), solution_status_t::SUCCESS);
  EXPECT_DOUBLE_EQ(solution.get_total_objective(), 8.);
  auto arrivals = cuopt::host_copy(
    solution.get_arrival_stamp().data(), solution.get_arrival_stamp().size(), handle.get_stream());
  EXPECT_EQ(arrivals, (std::vector<double>{0., 11., 28.}));
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

TEST(distance_matrices, unreachable_entries_are_clamped_in_distance_buffer_only)
{
  cpu_routing_problem_t problem;
  problem.num_locations     = 2;
  problem.fleet_size        = 1;
  problem.num_orders        = 1;
  problem.cost_matrices     = {{0, {0.f, 3.f, 5.f, 0.f}}};
  problem.distance_matrices = {{0, {0.f, std::numeric_limits<float>::infinity(), 1.e30f, 0.f}}};
  raft::handle_t handle;
  auto [model, storage] = problem.to_device(&handle);
  auto matrices         = detail::create_device_mdarray<float>(2, 1, 2, handle.get_stream());
  detail::fill_mdarray_from_data_model(matrices, model);
  auto values =
    cuopt::host_copy(matrices.buffer.data(), matrices.buffer.size(), handle.get_stream());
  EXPECT_EQ(values, (std::vector<float>{0.f, 3.f, 5.f, 0.f, 0.f, 1.e30f, 1.e30f, 0.f}));
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
  EXPECT_EQ((detail::get_cost_matrix_type_dim<int, float>(model)), 1);
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
