/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <routing/routing_details.hpp>
#include <routing/utilities/md_utils.hpp>
#include <utilities/macros.cuh>
#include <utilities/strided_span.hpp>

namespace cuopt {
namespace routing {
namespace detail {

/**
 * @brief Represents a distance tier with threshold and cost structure
 *
 * Example:
 * - Tier 1: threshold=100, fixed_cost=X, cost_per_unit=0
 * - Tier 2: threshold=200, fixed_cost=0, cost_per_unit=0.1
 * - Tier 3: threshold=max, fixed_cost=0, cost_per_unit=0.5
 */
template <typename f_t>
struct distance_tier_t {
  f_t threshold{0.0};      // Distance threshold (e.g., 100, 200)
  f_t fixed_cost{0.0};     // Fixed cost for this tier
  f_t cost_per_unit{0.0};  // Cost per km/unit for this tier

  bool operator==(distance_tier_t<f_t> const& rhs) const
  {
    return threshold == rhs.threshold && fixed_cost == rhs.fixed_cost &&
           cost_per_unit == rhs.cost_per_unit;
  }
};

template <typename f_t, bool is_device = true>
struct VehicleInfo {
  constexpr bool has_time_matrix() const
  {
    if (matrices.time_matrix_index != matrices.cost_matrix_index) { return true; }
    return matrices.extent[1] >= 2 && matrices.distance_matrix_index == matrices.cost_matrix_index;
  }

  HDI bool has_distance_tiers() const { return !distance_tiers.empty(); }

  HDI bool has_max_distance_constraint() const
  {
    return max_distance < std::numeric_limits<f_t>::max();
  }

  HDI bool uses_travel_distance() const
  {
    return has_distance_tiers() || has_max_distance_constraint();
  }

  HDI double compute_distance_cost(double travel_distance, double fallback_cost_distance) const
  {
    if (!has_distance_tiers()) { return fallback_cost_distance; }

    double tier_cost      = 0.0;
    double prev_threshold = 0.0;
    for (size_t i = 0; i < distance_tiers.size(); ++i) {
      const auto& tier = distance_tiers[i];
      if (travel_distance <= prev_threshold) { break; }
      const double upper   = tier.threshold;
      const double in_band = min(travel_distance, upper) - prev_threshold;
      if (in_band > 0.0) {
        if (tier.fixed_cost > 0.0) { tier_cost += tier.fixed_cost; }
        tier_cost += in_band * tier.cost_per_unit;
      }
      prev_threshold = upper;
      if (travel_distance <= upper) { break; }
    }

    return fallback_cost_distance + tier_cost;
  }
  bool operator==(VehicleInfo<f_t, is_device> const& rhs) const
  {
    if (distance_tiers.size() != rhs.distance_tiers.size()) { return false; }
    for (size_t i = 0; i < distance_tiers.size(); ++i) {
      if (!(distance_tiers[i] == rhs.distance_tiers[i])) { return false; }
    }

    return drop_return_trip == rhs.drop_return_trip && skip_first_trip == rhs.skip_first_trip &&
           type == rhs.type && order_service_times == rhs.order_service_times &&
           order_match == rhs.order_match && capacities == rhs.capacities &&
           break_durations == rhs.break_durations && break_earliest == rhs.break_earliest &&
           break_latest == rhs.break_latest && earliest == rhs.earliest && latest == rhs.latest &&
           start == rhs.start && end == rhs.end && max_cost == rhs.max_cost &&
           max_distance == rhs.max_distance &&
           max_time == rhs.max_time && fixed_cost == rhs.fixed_cost && priority == rhs.priority;
  }

  HDI int num_breaks() const { return break_durations.size(); }

  double get_average_distance() const
  {
    auto matrix  = matrices.get_distance_matrix(type);
    auto width   = matrices.extent[3];
    double sum   = 0.;
    size_t count = 0;

    for (size_t i = 0; i < width * width; ++i) {
      if (matrix[i] != std::numeric_limits<f_t>::max()) {
        sum += matrix[i];
        ++count;
      }
    }

    return count > 0 ? (sum / static_cast<double>(count)) : 0.0;
  }

  double get_average_cost() const
  {
    auto matrix  = matrices.get_cost_matrix(type);
    auto width   = matrices.extent[3];
    double sum   = 0.;
    size_t count = 0;

    for (size_t i = 0; i < width * width; ++i) {
      if (matrix[i] != std::numeric_limits<f_t>::max()) {
        sum += matrix[i];
        ++count;
      }
    }

    const double average_matrix_cost = count > 0 ? (sum / static_cast<double>(count)) : 0.0;
    if (distance_tiers.empty()) { return average_matrix_cost; }
    return compute_distance_cost(get_average_distance(), average_matrix_cost);
  }

  bool drop_return_trip = false;
  bool skip_first_trip  = false;
  uint8_t type{0};
  mdarray_view_t<f_t> matrices{};
  raft::span<int const, is_device> order_service_times{};
  raft::span<bool const, is_device> order_match{};
  cuopt::strided_span<cap_i_t const> capacities{};
  raft::span<int const, is_device> break_durations{};
  raft::span<int const, is_device> break_earliest{};
  raft::span<int const, is_device> break_latest{};
  int earliest{};
  int latest{};
  int start{};
  int end{};
  f_t max_cost = std::numeric_limits<f_t>::max();
  f_t max_distance = std::numeric_limits<f_t>::max();
  f_t max_time = std::numeric_limits<f_t>::max();
  f_t fixed_cost{};
  int priority{};

  // Distance tiers for tiered pricing based on total route distance
  // Tiers should be sorted by threshold in ascending order
  // Example: [{100, X, 0}, {200, 0, 0.1}, {INF, 0, 0.5}]
  raft::span<distance_tier_t<f_t> const, is_device> distance_tiers{};
};
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
