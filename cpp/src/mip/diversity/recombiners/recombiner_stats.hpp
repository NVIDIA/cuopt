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

#include <mip/problem/problem.cuh>

namespace cuopt::linear_programming::detail {

enum class recombiner_enum_t : int { BOUND_PROP = 0, FP, LINE_SEGMENT, SUB_MIP, SIZE };

constexpr std::array<recombiner_enum_t, 4> recombiner_types = {recombiner_enum_t::BOUND_PROP,
                                                               recombiner_enum_t::FP,
                                                               recombiner_enum_t::LINE_SEGMENT,
                                                               recombiner_enum_t::SUB_MIP};

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

  bool update_improve_stats(double cost_new, double cost_first, double cost_second)
  {
    bool is_better_than_both = false;
    if (cost_new < (std::min(cost_first, cost_second) - OBJECTIVE_EPSILON)) {
      ++better_than_both;
      is_better_than_both = true;
    }
    if (cost_new < (std::max(cost_first, cost_second) - OBJECTIVE_EPSILON)) ++better_than_one;
    return is_better_than_both;
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
  static constexpr std::array recombiner_labels = {"BOUND_PROP", "FP", "LINE_SEGMENT", "SUB_MIP"};

  std::array<recombine_stats, recombiner_count> stats;

  static_assert(recombiner_labels.size() == (size_t)recombiner_enum_t::SIZE,
                "Mismatch between names and enums");

  // enum of the last attempted recombiner
  std::optional<recombiner_enum_t> last_attempt;
  double last_recombiner_time;
  std::chrono::high_resolution_clock::time_point last_recombiner_start_time;

  void start_recombiner_time()
  {
    last_recombiner_start_time = std::chrono::high_resolution_clock::now();
  }
  void stop_recombiner_time()
  {
    last_recombiner_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::high_resolution_clock::now() - last_recombiner_start_time)
                             .count();
  }

  double get_last_recombiner_time() { return last_recombiner_time; }

  void reset()
  {
    for (size_t i = 0; i < recombiner_count; ++i) {
      stats[i].reset();
    }
    last_attempt.reset();
  }

  recombiner_enum_t get_last_attempt() { return last_attempt.value(); }

  void add_attempt(recombiner_enum_t r)
  {
    last_attempt = r;
    stats[static_cast<int>(r)].add_attempt();
  }

  void add_success() { stats[static_cast<int>(last_attempt.value())].add_success(); }

  bool update_improve_stats(double cost_new, double cost_first, double cost_second)
  {
    return stats[static_cast<int>(last_attempt.value())].update_improve_stats(
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
