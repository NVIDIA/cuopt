/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "fsmvrptwsc_parser.hpp"

#include <routing/utilities/check_constraints.hpp>
#include <routing/vehicle_info.hpp>

#include <cuopt/routing/solve.hpp>
#include <utilities/base_fixture.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

namespace {

struct fsmvrptwsc_params_t {
  std::string small_file;
  std::string instance_name;
  float reference_cost{};
  float max_relative_gap{};
};

bool is_absolute_path(std::string const& path)
{
  return !path.empty() &&
         (path[0] == '/' || path[0] == '\\' || (path.size() > 1 && path[1] == ':'));
}

std::string join_path(std::string const& base, std::string const& path)
{
  if (base.empty()) { return path; }
  if (base.back() == '/' || base.back() == '\\') { return base + path; }
  return base + "/" + path;
}

std::string resolve_ref_path(std::string const& ref_file)
{
  if (is_absolute_path(ref_file)) { return ref_file; }
  auto const cuopt_home = cuopt::test::get_cuopt_home();
  return cuopt_home.empty() ? ref_file : join_path(cuopt_home, ref_file);
}

std::string resolve_dataset_path(std::string const& dataset_file)
{
  if (is_absolute_path(dataset_file)) { return dataset_file; }

  auto dataset_root     = cuopt::test::get_rapids_dataset_root_dir();
  auto const cuopt_home = cuopt::test::get_cuopt_home();
  if (!is_absolute_path(dataset_root) && !cuopt_home.empty()) {
    dataset_root = join_path(cuopt_home, dataset_root);
  }

  return join_path(dataset_root, dataset_file);
}

std::vector<fsmvrptwsc_params_t> read_fsmvrptwsc_tests(std::string const& ref_file)
{
  std::ifstream infile(resolve_ref_path(ref_file));
  if (!infile.is_open()) { throw std::runtime_error("Ref file cannot be opened: " + ref_file); }

  std::vector<fsmvrptwsc_params_t> params;
  for (std::string line; getline(infile, line);) {
    if (line.empty()) { continue; }
    auto tokens = cuopt::test::split(line, ',');
    if (tokens.size() != 4) { throw std::runtime_error("Invalid FSMVRPTWSC ref line: " + line); }
    params.push_back(
      {resolve_dataset_path(tokens[0]), tokens[1], std::stof(tokens[2]), std::stof(tokens[3])});
  }
  return params;
}

}  // namespace

class fsmvrptwsc_small_test_t : public ::testing::TestWithParam<fsmvrptwsc_params_t> {};

TEST(fsmvrptwsc_parser, range_starts_enter_the_next_tier)
{
  std::istringstream input(
    "boundary 1 1 3 "
    "0 1 1 0 "              // Distance matrix.
    "0 1 1 0 "              // Transit-time matrix.
    "0 100 0 0 0 100 0 1 "  // Depot and order windows, service times and demands.
    "10 0 40 70 50 58 2");  // Capacity, range starts and costs.
  auto instance = read_one_instance(input);
  ASSERT_EQ(instance.tier_thresholds.size(), 3);
  std::vector<detail::distance_tier_t<float>> tiers;
  for (size_t i = 0; i < instance.tier_thresholds.size(); ++i) {
    tiers.push_back(
      {instance.tier_thresholds[i], instance.tier_fixed_costs[i], instance.tier_costs_per_unit[i]});
  }
  detail::VehicleInfo<float, false> vehicle;
  vehicle.distance_tiers =
    raft::span<detail::distance_tier_t<float> const, false>(tiers.data(), tiers.size());
  EXPECT_DOUBLE_EQ(vehicle.compute_distance_cost(instance.tier_thresholds[0], 0.), 50.);
  EXPECT_DOUBLE_EQ(vehicle.compute_distance_cost(40., 0.), 58.);
  EXPECT_EQ(vehicle.find_distance_tier(40.), 1);
  EXPECT_EQ(vehicle.find_distance_tier(70.), 2);
  EXPECT_FLOAT_EQ(instance.tier_thresholds.back(), std::numeric_limits<float>::max());
  EXPECT_NEAR(vehicle.compute_distance_cost(75., 0.), 68., 1.e-4);
}

TEST(fsmvrptwsc_parser, missing_instance_throws)
{
  auto const params = read_fsmvrptwsc_tests("datasets/ref/fsmvrptwsc_small.txt");
  ASSERT_FALSE(params.empty());
  EXPECT_THROW(load_small_instance(params.front().small_file, "missing-instance"),
               std::runtime_error);
}

TEST_P(fsmvrptwsc_small_test_t, solves_small_step_cost_instance)
{
  auto const param = GetParam();
  auto instance    = load_small_instance(param.small_file, param.instance_name);

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto zero_cost_matrix = std::vector<float>(instance.distance_matrix.size(), 0.0f);

  auto d_cost_matrix         = cuopt::device_copy(zero_cost_matrix, stream);
  auto d_distance_matrix     = cuopt::device_copy(instance.distance_matrix, stream);
  auto d_transit_time_matrix = cuopt::device_copy(instance.transit_time_matrix, stream);
  auto d_order_locations     = cuopt::device_copy(instance.order_locations, stream);
  auto d_earliest            = cuopt::device_copy(instance.earliest, stream);
  auto d_latest              = cuopt::device_copy(instance.latest, stream);
  auto d_service_times       = cuopt::device_copy(instance.service_times, stream);
  auto d_demands             = cuopt::device_copy(instance.demand, stream);
  auto d_vehicle_earliest    = cuopt::device_copy(instance.vehicle_earliest, stream);
  auto d_vehicle_latest      = cuopt::device_copy(instance.vehicle_latest, stream);
  auto d_capacities          = cuopt::device_copy(instance.capacities, stream);
  auto d_vehicle_types       = cuopt::device_copy(instance.vehicle_types, stream);
  auto d_tier_thresholds     = cuopt::device_copy(instance.tier_thresholds, stream);
  auto d_tier_fixed_costs    = cuopt::device_copy(instance.tier_fixed_costs, stream);
  auto d_tier_costs_per_unit = cuopt::device_copy(instance.tier_costs_per_unit, stream);
  auto d_tier_offsets        = cuopt::device_copy(instance.tier_offsets, stream);
  handle.sync_stream();

  cuopt::routing::data_model_view_t<int, float> data_model(
    &handle, instance.n_clients + 1, instance.n_vehicles, instance.n_clients);
  for (int type = 0; type < instance.n_vehicle_types; ++type) {
    data_model.add_cost_matrix(d_cost_matrix.data(), static_cast<uint8_t>(type));
    data_model.add_distance_matrix(d_distance_matrix.data(), static_cast<uint8_t>(type));
    data_model.add_transit_time_matrix(d_transit_time_matrix.data(), static_cast<uint8_t>(type));
  }
  data_model.set_order_locations(d_order_locations.data());
  data_model.set_order_time_windows(d_earliest.data(), d_latest.data(), false);
  data_model.set_order_service_times(d_service_times.data(), -1, false);
  data_model.set_vehicle_time_windows(d_vehicle_earliest.data(), d_vehicle_latest.data(), false);
  data_model.set_vehicle_types(d_vehicle_types.data(), false);
  data_model.add_capacity_dimension("demand", d_demands.data(), d_capacities.data(), false);
  data_model.set_vehicle_distance_tiers(d_tier_thresholds.data(),
                                        d_tier_fixed_costs.data(),
                                        d_tier_costs_per_unit.data(),
                                        d_tier_offsets.data(),
                                        static_cast<int>(instance.tier_thresholds.size()));

  cuopt::routing::solver_settings_t<int, float> settings;
  // Use longer time limit for larger real instances.
  auto time_limit = (instance.n_clients > 50) ? 300.0f : 5.0f;
  settings.set_time_limit(time_limit);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  auto host_route = cuopt::routing::host_assignment_t<int>(routing_solution);
  check_route(data_model, host_route);

  auto const objective = routing_solution.get_total_objective();
  auto const max_cost  = param.reference_cost * (1.0f + param.max_relative_gap);

  EXPECT_LE(objective, max_cost) << "FSMVRPTWSC gap exceeded for " << param.instance_name;
}

INSTANTIATE_TEST_SUITE_P(
  small,
  fsmvrptwsc_small_test_t,
  ::testing::ValuesIn(read_fsmvrptwsc_tests("datasets/ref/fsmvrptwsc_small.txt")));

}  // namespace test
}  // namespace routing
}  // namespace cuopt
