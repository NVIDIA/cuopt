/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

namespace cuopt::linear_programming::detail {

enum recombiner_enum_t : int { BOUND_PROP = 0, FP, LINE_SEGMENT, SIZE };

struct recombine_stats {
  int attempts;
  int success;
  int better_than_one;
  int better_than_both;

  void reset()
  {
    attempts         = 0;
    success          = 0;
    better_than_one  = 0;
    better_than_both = 0;
  }

  void add_success() { ++success; }

  void update_improve_stats(double cost_new, double cost_first, double cost_second)
  {
    if (cost_new < (std::min(cost_first, cost_second) - OBJECTIVE_EPSILON)) ++better_than_both;
    if (cost_new < (std::max(cost_first, cost_second) - OBJECTIVE_EPSILON)) ++better_than_one;
  }

  void add_attempt() { ++attempts; }

  void print([[maybe_unused]] const char* recombiner_name)
  {
    CUOPT_LOG_DEBUG("%s : (better_than_one: %d better_than_both: %d success: %d attempts: %d)\t",
                    recombiner_name,
                    better_than_one,
                    better_than_both,
                    success,
                    attempts);
  }
};

struct all_recombine_stats {
  static constexpr size_t recombiner_count      = static_cast<int>(recombiner_enum_t::SIZE);
  static constexpr std::array recombiner_labels = {"BOUND_PROP", "FP", "LINE_SEGMENT"};

  std::array<recombine_stats, recombiner_count> stats;

  static_assert(recombiner_labels.size() == (size_t)recombiner_enum_t::SIZE,
                "Mismatch between names and enums");

  // enum of the last attempted recombiner
  std::optional<recombiner_enum_t> last_attempt;

  void reset()
  {
    for (size_t i = 0; i < recombiner_count; ++i) {
      stats[i].reset();
    }
    last_attempt.reset();
  }

  void add_attempt(recombiner_enum_t r)
  {
    last_attempt = r;
    stats[static_cast<int>(r)].add_attempt();
  }

  void add_success() { stats[static_cast<int>(last_attempt.value())].add_success(); }

  void update_improve_stats(double cost_new, double cost_first, double cost_second)
  {
    stats[static_cast<int>(last_attempt.value())].update_improve_stats(
      cost_new, cost_first, cost_second);
  }

  void print()
  {
    CUOPT_LOG_DEBUG("Recombiner stats: ");
    for (size_t i = 0; i < recombiner_count; ++i) {
      stats[i].print(recombiner_labels[i]);
    }
  }
};

}  // namespace cuopt::linear_programming::detail