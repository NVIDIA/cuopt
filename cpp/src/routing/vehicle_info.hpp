/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <routing/routing_details.hpp>
#include <routing/utilities/md_utils.hpp>
#include <utilities/macros.cuh>
#include <utilities/strided_span.hpp>

namespace cuopt {
namespace routing {
namespace detail {

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
};
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
