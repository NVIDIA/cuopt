/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <cuopt/routing/solve.hpp>
#include <routing/node/distance_node.cuh>
#include <routing/problem/problem.cuh>
#include <routing/utilities/md_utils.hpp>
#include <utilities/copy_helpers.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

namespace {

struct tier_buffers_t {
  rmm::device_uvector<float> thresholds;
  rmm::device_uvector<float> fixed_costs;
  rmm::device_uvector<float> costs_per_unit;
  rmm::device_uvector<int> offsets;

  explicit tier_buffers_t(rmm::cuda_stream_view stream)
    : thresholds(0, stream), fixed_costs(0, stream), costs_per_unit(0, stream), offsets(0, stream)
  {
  }
};

tier_buffers_t make_tier_buffers(rmm::cuda_stream_view stream,
                                 std::vector<float> const& thresholds,
                                 std::vector<float> const& fixed_costs,
                                 std::vector<float> const& costs_per_unit,
                                 std::vector<int> const& tier_offsets)
{
  tier_buffers_t buffers(stream);
  buffers.thresholds     = cuopt::device_copy(thresholds, stream);
  buffers.fixed_costs    = cuopt::device_copy(fixed_costs, stream);
  buffers.costs_per_unit = cuopt::device_copy(costs_per_unit, stream);
  buffers.offsets        = cuopt::device_copy(tier_offsets, stream);
  return buffers;
}

tier_buffers_t make_uniform_two_band_tiers(rmm::cuda_stream_view stream,
                                           int nvehicles,
                                           float threshold,
                                           float overflow_cost_per_unit)
{
  std::vector<float> thresholds;
  std::vector<float> fixed_costs;
  std::vector<float> costs_per_unit;
  std::vector<int> tier_offsets{0};

  thresholds.reserve(2 * nvehicles);
  fixed_costs.reserve(2 * nvehicles);
  costs_per_unit.reserve(2 * nvehicles);

  for (int vehicle_id = 0; vehicle_id < nvehicles; ++vehicle_id) {
    thresholds.push_back(threshold);
    fixed_costs.push_back(0.f);
    costs_per_unit.push_back(0.f);

    thresholds.push_back(1.0e9f);
    fixed_costs.push_back(0.f);
    costs_per_unit.push_back(overflow_cost_per_unit);

    tier_offsets.push_back(static_cast<int>(thresholds.size()));
  }

  return make_tier_buffers(stream, thresholds, fixed_costs, costs_per_unit, tier_offsets);
}

void set_vehicle_distance_tiers(cuopt::routing::data_model_view_t<int, float>& data_model,
                                tier_buffers_t const& buffers)
{
  data_model.set_vehicle_distance_tiers(buffers.thresholds.data(),
                                        buffers.fixed_costs.data(),
                                        buffers.costs_per_unit.data(),
                                        buffers.offsets.data(),
                                        static_cast<int>(buffers.thresholds.size()));
}

}  // namespace

TEST(distance_tiers_separate_distance, solver_uses_separate_distance_matrix_for_tiered_costs)
{
  constexpr int nlocations = 2;
  constexpr int norders    = 1;
  constexpr int nvehicles  = 1;

  std::vector<float> cost_matrix = {
    0.f, 1.f,
    1.f, 0.f,
  };
  std::vector<float> distance_matrix = {
    0.f, 5.f,
    5.f, 0.f,
  };
  std::vector<int> order_locations = {1};
  std::vector<int> demands         = {1};
  std::vector<int> capacities      = {1};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto d_cost_matrix      = cuopt::device_copy(cost_matrix, stream);
  auto d_distance_matrix  = cuopt::device_copy(distance_matrix, stream);
  auto d_order_locations  = cuopt::device_copy(order_locations, stream);
  auto d_demands          = cuopt::device_copy(demands, stream);
  auto d_capacities       = cuopt::device_copy(capacities, stream);
  auto tier_buffers       = make_uniform_two_band_tiers(stream, nvehicles, 8.0f, 3.0f);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(d_cost_matrix.data());
  data_model.add_distance_matrix(d_distance_matrix.data());
  data_model.set_order_locations(d_order_locations.data());
  data_model.add_capacity_dimension("demand", d_demands.data(), d_capacities.data());
  set_vehicle_distance_tiers(data_model, tier_buffers);

  auto routing_solution = cuopt::routing::solve(data_model);
  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  ASSERT_EQ(routing_solution.get_vehicle_count(), 1);
  ASSERT_NEAR(routing_solution.get_total_objective(), 8.0f, 1e-5);
}

TEST(distance_tiers_separate_distance, solver_uses_tier_fixed_cost_in_objective)
{
  constexpr int nlocations = 2;
  constexpr int norders    = 1;
  constexpr int nvehicles  = 1;

  std::vector<float> cost_matrix = {
    0.f, 1.f,
    1.f, 0.f,
  };
  std::vector<float> distance_matrix = {
    0.f, 5.f,
    5.f, 0.f,
  };
  std::vector<int> order_locations = {1};
  std::vector<int> demands         = {1};
  std::vector<int> capacities      = {1};
  std::vector<float> thresholds    = {8.f, 1.0e9f};
  std::vector<float> fixed_costs   = {0.f, 7.f};
  std::vector<float> costs_per_unit = {0.f, 3.f};
  std::vector<int> tier_offsets    = {0, 2};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto d_cost_matrix      = cuopt::device_copy(cost_matrix, stream);
  auto d_distance_matrix  = cuopt::device_copy(distance_matrix, stream);
  auto d_order_locations  = cuopt::device_copy(order_locations, stream);
  auto d_demands          = cuopt::device_copy(demands, stream);
  auto d_capacities       = cuopt::device_copy(capacities, stream);
  auto tier_buffers =
    make_tier_buffers(stream, thresholds, fixed_costs, costs_per_unit, tier_offsets);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(d_cost_matrix.data());
  data_model.add_distance_matrix(d_distance_matrix.data());
  data_model.set_order_locations(d_order_locations.data());
  data_model.add_capacity_dimension("demand", d_demands.data(), d_capacities.data());
  set_vehicle_distance_tiers(data_model, tier_buffers);

  auto routing_solution = cuopt::routing::solve(data_model);
  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  ASSERT_NEAR(routing_solution.get_total_objective(), 15.0f, 1e-5);
}

TEST(distance_tiers_separate_distance, compute_distance_cost_treats_threshold_as_inclusive_upper_bound)
{
  using vehicle_info_t = cuopt::routing::detail::VehicleInfo<float, false>;
  using distance_tier_t = cuopt::routing::detail::distance_tier_t<float>;

  std::vector<distance_tier_t> tiers = {{10.f, 0.f, 0.f}, {1.0e9f, 5.f, 3.f}};
  vehicle_info_t vehicle_info{};
  vehicle_info.distance_tiers = raft::span<distance_tier_t const, false>(tiers.data(), tiers.size());

  ASSERT_NEAR(vehicle_info.compute_distance_cost(10.f, 2.f), 2.f, 1e-5);
  ASSERT_NEAR(vehicle_info.compute_distance_cost(11.f, 2.f), 10.f, 1e-5);
}

TEST(distance_tiers_separate_distance, compute_distance_cost_accumulates_fixed_costs_across_tiers)
{
  using vehicle_info_t = cuopt::routing::detail::VehicleInfo<float, false>;
  using distance_tier_t = cuopt::routing::detail::distance_tier_t<float>;

  std::vector<distance_tier_t> tiers = {
    {5.f, 4.f, 0.f}, {10.f, 7.f, 0.f}, {1.0e9f, 0.f, 2.f}};
  vehicle_info_t vehicle_info{};
  vehicle_info.distance_tiers = raft::span<distance_tier_t const, false>(tiers.data(), tiers.size());

  ASSERT_NEAR(vehicle_info.compute_distance_cost(12.f, 3.f), 18.f, 1e-5);
}

TEST(distance_tiers_separate_distance, solver_applies_heterogeneous_tier_offsets_per_vehicle)
{
  constexpr int nlocations = 3;
  constexpr int norders    = 2;
  constexpr int nvehicles  = 2;

  std::vector<float> cost_matrix = {
    0.f, 1.f, 1.f,
    1.f, 0.f, 1.f,
    1.f, 1.f, 0.f,
  };
  std::vector<float> distance_matrix = {
    0.f, 5.f, 2.f,
    5.f, 0.f, 1.f,
    2.f, 1.f, 0.f,
  };
  std::vector<int> order_locations = {1, 2};
  std::vector<int> demands         = {1, 1};
  std::vector<int> capacities      = {1, 1};
  std::vector<int> order_zero_allowed_vehicles = {0};
  std::vector<int> order_one_allowed_vehicles  = {1};
  std::vector<float> thresholds = {8.f, 1.0e9f, 5.f, 6.f, 1.0e9f};
  std::vector<float> fixed_costs = {0.f, 0.f, 5.f, 0.f, 0.f};
  std::vector<float> costs_per_unit = {0.f, 3.f, 0.f, 4.f, 9.f};
  std::vector<int> tier_offsets = {0, 2, 5};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto d_cost_matrix      = cuopt::device_copy(cost_matrix, stream);
  auto d_distance_matrix  = cuopt::device_copy(distance_matrix, stream);
  auto d_order_locations  = cuopt::device_copy(order_locations, stream);
  auto d_demands          = cuopt::device_copy(demands, stream);
  auto d_capacities       = cuopt::device_copy(capacities, stream);
  auto d_order_zero_match = cuopt::device_copy(order_zero_allowed_vehicles, stream);
  auto d_order_one_match  = cuopt::device_copy(order_one_allowed_vehicles, stream);
  auto tier_buffers =
    make_tier_buffers(stream, thresholds, fixed_costs, costs_per_unit, tier_offsets);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(d_cost_matrix.data());
  data_model.add_distance_matrix(d_distance_matrix.data());
  data_model.set_order_locations(d_order_locations.data());
  data_model.add_capacity_dimension("demand", d_demands.data(), d_capacities.data());
  data_model.add_order_vehicle_match(0, d_order_zero_match.data(), 1);
  data_model.add_order_vehicle_match(1, d_order_one_match.data(), 1);
  data_model.set_min_vehicles(2);
  set_vehicle_distance_tiers(data_model, tier_buffers);

  auto routing_solution = cuopt::routing::solve(data_model);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  ASSERT_EQ(routing_solution.get_vehicle_count(), 2);
  ASSERT_NEAR(routing_solution.get_total_objective(), 15.0f, 1e-5);

  auto node_types_host = cuopt::host_copy(routing_solution.get_node_types(), stream);
  auto truck_id_host   = cuopt::host_copy(routing_solution.get_truck_id(), stream);

  std::vector<int> non_depot_vehicles;
  for (size_t i = 0; i < node_types_host.size(); ++i) {
    if (node_types_host[i] != static_cast<int>(cuopt::routing::node_type_t::DEPOT)) {
      non_depot_vehicles.push_back(truck_id_host[i]);
    }
  }

  std::sort(non_depot_vehicles.begin(), non_depot_vehicles.end());
  ASSERT_EQ(non_depot_vehicles.size(), 2);
  EXPECT_EQ(non_depot_vehicles[0], 0);
  EXPECT_EQ(non_depot_vehicles[1], 1);
}

TEST(distance_tiers_separate_distance,
     solver_vehicle_choice_changes_when_tiers_use_separate_distance_matrix)
{
  constexpr int nlocations = 2;
  constexpr int norders    = 1;
  constexpr int nvehicles  = 2;

  std::vector<float> cost_matrix_type_zero = {
    0.f, 1.f,
    1.f, 0.f,
  };
  std::vector<float> cost_matrix_type_one = {
    0.f, 2.f,
    2.f, 0.f,
  };
  std::vector<float> distance_matrix_type_zero = {
    0.f, 5.f,
    5.f, 0.f,
  };
  std::vector<float> distance_matrix_type_one = {
    0.f, 1.f,
    1.f, 0.f,
  };
  std::vector<uint8_t> vehicle_types = {0, 1};
  std::vector<int> order_locations   = {1};
  std::vector<int> demands           = {1};
  std::vector<int> capacities        = {1, 1};
  std::vector<float> thresholds      = {4.f, 1.0e9f, 4.f, 1.0e9f};
  std::vector<float> fixed_costs     = {0.f, 0.f, 0.f, 0.f};
  std::vector<float> costs_per_unit  = {0.f, 10.f, 0.f, 10.f};
  std::vector<int> tier_offsets      = {0, 2, 4};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto d_cost_matrix_type_zero     = cuopt::device_copy(cost_matrix_type_zero, stream);
  auto d_cost_matrix_type_one      = cuopt::device_copy(cost_matrix_type_one, stream);
  auto d_distance_matrix_type_zero = cuopt::device_copy(distance_matrix_type_zero, stream);
  auto d_distance_matrix_type_one  = cuopt::device_copy(distance_matrix_type_one, stream);
  auto d_vehicle_types             = cuopt::device_copy(vehicle_types, stream);
  auto d_order_locations           = cuopt::device_copy(order_locations, stream);
  auto d_demands                   = cuopt::device_copy(demands, stream);
  auto d_capacities                = cuopt::device_copy(capacities, stream);
  auto tier_buffers =
    make_tier_buffers(stream, thresholds, fixed_costs, costs_per_unit, tier_offsets);

  cuopt::routing::data_model_view_t<int, float> cost_only_data_model(
    &handle, nlocations, nvehicles, norders);
  cost_only_data_model.add_cost_matrix(d_cost_matrix_type_zero.data(), 0);
  cost_only_data_model.add_cost_matrix(d_cost_matrix_type_one.data(), 1);
  cost_only_data_model.set_vehicle_types(d_vehicle_types.data());
  cost_only_data_model.set_order_locations(d_order_locations.data());
  cost_only_data_model.add_capacity_dimension("demand", d_demands.data(), d_capacities.data());

  auto cost_only_solution = cuopt::routing::solve(cost_only_data_model);
  handle.sync_stream();
  ASSERT_EQ(cost_only_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  ASSERT_EQ(cost_only_solution.get_vehicle_count(), 1);
  ASSERT_NEAR(cost_only_solution.get_total_objective(), 2.0f, 1e-5);

  auto cost_only_node_types = cuopt::host_copy(cost_only_solution.get_node_types(), stream);
  auto cost_only_truck_ids  = cuopt::host_copy(cost_only_solution.get_truck_id(), stream);
  int cost_only_serving_vehicle = -1;
  int cost_only_non_depot_count = 0;
  for (size_t i = 0; i < cost_only_node_types.size(); ++i) {
    if (cost_only_node_types[i] != static_cast<int>(cuopt::routing::node_type_t::DEPOT)) {
      cost_only_serving_vehicle = cost_only_truck_ids[i];
      ++cost_only_non_depot_count;
    }
  }
  ASSERT_EQ(cost_only_non_depot_count, 1);
  ASSERT_EQ(cost_only_serving_vehicle, 0);

  cuopt::routing::data_model_view_t<int, float> tiered_data_model(
    &handle, nlocations, nvehicles, norders);
  tiered_data_model.add_cost_matrix(d_cost_matrix_type_zero.data(), 0);
  tiered_data_model.add_cost_matrix(d_cost_matrix_type_one.data(), 1);
  tiered_data_model.add_distance_matrix(d_distance_matrix_type_zero.data(), 0);
  tiered_data_model.add_distance_matrix(d_distance_matrix_type_one.data(), 1);
  tiered_data_model.set_vehicle_types(d_vehicle_types.data());
  tiered_data_model.set_order_locations(d_order_locations.data());
  tiered_data_model.add_capacity_dimension("demand", d_demands.data(), d_capacities.data());
  set_vehicle_distance_tiers(tiered_data_model, tier_buffers);

  auto tiered_solution = cuopt::routing::solve(tiered_data_model);
  handle.sync_stream();
  ASSERT_EQ(tiered_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
  ASSERT_EQ(tiered_solution.get_vehicle_count(), 1);
  ASSERT_NEAR(tiered_solution.get_total_objective(), 4.0f, 1e-5);

  auto tiered_node_types = cuopt::host_copy(tiered_solution.get_node_types(), stream);
  auto tiered_truck_ids  = cuopt::host_copy(tiered_solution.get_truck_id(), stream);
  int tiered_serving_vehicle = -1;
  int tiered_non_depot_count = 0;
  for (size_t i = 0; i < tiered_node_types.size(); ++i) {
    if (tiered_node_types[i] != static_cast<int>(cuopt::routing::node_type_t::DEPOT)) {
      tiered_serving_vehicle = tiered_truck_ids[i];
      ++tiered_non_depot_count;
    }
  }
  ASSERT_EQ(tiered_non_depot_count, 1);
  ASSERT_EQ(tiered_serving_vehicle, 1);
}

TEST(distance_tiers_separate_distance,
     solver_respects_vehicle_max_distance_from_separate_distance_matrix)
{
  constexpr int nlocations = 2;
  constexpr int norders    = 1;
  constexpr int nvehicles  = 1;

  std::vector<float> cost_matrix = {
    0.f, 1.f,
    1.f, 0.f,
  };
  std::vector<float> distance_matrix = {
    0.f, 5.f,
    5.f, 0.f,
  };
  std::vector<int> order_locations = {1};
  std::vector<int> demands         = {1};
  std::vector<int> capacities      = {1};
  std::vector<float> max_distances = {9.f};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto d_cost_matrix      = cuopt::device_copy(cost_matrix, stream);
  auto d_distance_matrix  = cuopt::device_copy(distance_matrix, stream);
  auto d_order_locations  = cuopt::device_copy(order_locations, stream);
  auto d_demands          = cuopt::device_copy(demands, stream);
  auto d_capacities       = cuopt::device_copy(capacities, stream);
  auto d_max_distances    = cuopt::device_copy(max_distances, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(d_cost_matrix.data());
  data_model.add_distance_matrix(d_distance_matrix.data());
  data_model.set_order_locations(d_order_locations.data());
  data_model.add_capacity_dimension("demand", d_demands.data(), d_capacities.data());
  data_model.set_vehicle_max_distances(d_max_distances.data());

  auto routing_solution = cuopt::routing::solve(data_model);
  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::INFEASIBLE);
}

TEST(distance_tiers_separate_distance, distance_node_combine_respects_tiered_max_cost)
{
  using vehicle_info_t = cuopt::routing::detail::VehicleInfo<float, false>;
  using distance_node_t = cuopt::routing::detail::distance_node_t<int, float>;
  using distance_tier_t = cuopt::routing::detail::distance_tier_t<float>;

  std::vector<distance_tier_t> tiers = {{8.f, 0.f, 0.f}, {1.0e9f, 0.f, 3.f}};
  vehicle_info_t vehicle_info{};
  vehicle_info.max_distance   = 100.f;
  vehicle_info.max_cost       = 5.f;
  vehicle_info.distance_tiers = raft::span<distance_tier_t const, false>(tiers.data(), tiers.size());

  distance_node_t prev{};
  prev.distance_forward        = 1.f;
  prev.travel_distance_forward = 5.f;

  distance_node_t next{};
  next.distance_backward        = 1.f;
  next.travel_distance_backward = 5.f;

  const double combined_excess = distance_node_t::combine(prev, next, vehicle_info, 0.f, 0.f);
  ASSERT_NEAR(combined_excess, 3.f, 1e-5);
}

TEST(distance_tiers_separate_distance, viable_neighbor_score_uses_tiers_and_cost_matrix)
{
  using problem_t = cuopt::routing::detail::problem_t<int, float>;
  using vehicle_info_t = cuopt::routing::detail::VehicleInfo<float, false>;
  using distance_tier_t = cuopt::routing::detail::distance_tier_t<float>;

  std::vector<float> cost_matrix = {
    0.f, 1.f, 4.f,
    1.f, 0.f, 0.f,
    4.f, 0.f, 0.f,
  };
  std::vector<float> distance_matrix = {
    0.f, 5.f, 1.f,
    5.f, 0.f, 0.f,
    1.f, 0.f, 0.f,
  };
  std::vector<distance_tier_t> tiers = {{2.f, 0.f, 0.f}, {1.0e9f, 0.f, 10.f}};

  cuopt::routing::h_mdarray_t<float> matrices({1, 3, 3, 3});
  matrices.cost_matrix_index = 0;
  matrices.distance_matrix_index = 1;
  matrices.time_matrix_index = 2;
  std::copy(cost_matrix.begin(), cost_matrix.end(), matrices.get_cost_matrix(0, 0));
  std::copy(distance_matrix.begin(), distance_matrix.end(), matrices.get_cost_matrix(0, 1));
  std::copy(distance_matrix.begin(), distance_matrix.end(), matrices.get_cost_matrix(0, 2));

  vehicle_info_t vehicle_info{};
  vehicle_info.type           = 0;
  vehicle_info.matrices       = matrices.view();
  vehicle_info.distance_tiers = raft::span<distance_tier_t const, false>(tiers.data(), tiers.size());

  const auto from = cuopt::routing::detail::NodeInfo<int>(
    0, 0, cuopt::routing::node_type_t::PICKUP);
  const auto near_by_distance =
    cuopt::routing::detail::NodeInfo<int>(1, 1, cuopt::routing::node_type_t::PICKUP);
  const auto near_by_cost =
    cuopt::routing::detail::NodeInfo<int>(2, 2, cuopt::routing::node_type_t::PICKUP);

  const double distance_neighbor_score =
    problem_t::compute_viable_neighbor_score(from, near_by_distance, vehicle_info);
  const double cost_neighbor_score =
    problem_t::compute_viable_neighbor_score(from, near_by_cost, vehicle_info);

  ASSERT_NEAR(distance_neighbor_score, 31.f, 1e-5);
  ASSERT_NEAR(cost_neighbor_score, 4.f, 1e-5);
  ASSERT_LT(cost_neighbor_score, distance_neighbor_score);
}

TEST(distance_tiers_separate_distance,
     problem_uses_zero_travel_distance_and_preserves_host_cost_matrix_without_distance_matrix)
{
  using problem_t = cuopt::routing::detail::problem_t<int, float>;

  constexpr int nlocations = 2;
  constexpr int norders    = 1;
  constexpr int nvehicles  = 1;

  std::vector<float> cost_matrix = {
    0.f, 7.f,
    3.f, 0.f,
  };
  std::vector<int> order_locations = {1};
  std::vector<int> demands         = {1};
  std::vector<int> capacities      = {1};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto d_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto d_order_locations = cuopt::device_copy(order_locations, stream);
  auto d_demands         = cuopt::device_copy(demands, stream);
  auto d_capacities      = cuopt::device_copy(capacities, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(d_cost_matrix.data());
  data_model.set_order_locations(d_order_locations.data());
  data_model.add_capacity_dimension("demand", d_demands.data(), d_capacities.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  problem_t problem(data_model, settings);

  const auto depot = problem.get_start_depot_node_info(0);
  const auto order = cuopt::routing::detail::NodeInfo<int>(0, 1, cuopt::routing::node_type_t::PICKUP);

  ASSERT_NEAR(problem.distance_between(depot, order, 0), 0.f, 1e-5);
  ASSERT_NEAR(problem.cost_between(depot, order, 0), 7.f, 1e-5);
  ASSERT_NEAR(problem.cost_between(order, depot, 0), 3.f, 1e-5);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
