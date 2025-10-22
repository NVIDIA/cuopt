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
};

template <typename f_t, bool is_device = true>
struct VehicleInfo {
  constexpr bool has_time_matrix() const { return matrices.extent[1] > 1; }

  bool operator==(VehicleInfo<f_t, is_device> const& rhs) const
  {
    return drop_return_trip == rhs.drop_return_trip && skip_first_trip == rhs.skip_first_trip &&
           type == rhs.type && order_service_times == rhs.order_service_times &&
           order_match == rhs.order_match && capacities == rhs.capacities &&
           break_durations == rhs.break_durations && break_earliest == rhs.break_earliest &&
           break_latest == rhs.break_latest && earliest == rhs.earliest && latest == rhs.latest &&
           start == rhs.start && end == rhs.end && max_cost == rhs.max_cost &&
           max_time == rhs.max_time && fixed_cost == rhs.fixed_cost && priority == rhs.priority;
  }

  HDI int num_breaks() const { return break_durations.size(); }

  double get_average_cost() const
  {
    auto matrix     = matrices.get_cost_matrix(type);
    auto width      = matrices.extent[3];
    double avg_cost = 0.;

    for (size_t i = 0; i < width * width; ++i) {
      if (matrix[i] != std::numeric_limits<f_t>::max()) { avg_cost += matrix[i]; }
    }

    return avg_cost / (width * width);
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
