/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/diversity/population.cuh>
#include <mip_heuristics/feasibility_jump/cpu/state.hpp>

#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Host-side LNS seed source. CPU workers read it without touching device memory or the
// population locks.
template <typename i_t, typename f_t>
class lns_population_feed_t {
 public:
  explicit lns_population_feed_t(population_t<i_t, f_t>& population) : population_(population)
  {
    population_.set_feasible_solution_callback(
      [this](const std::vector<f_t>& assignment, f_t objective, f_t user_objective) {
        best_.publish(objective, user_objective, assignment);
      });
  }
  ~lns_population_feed_t() { population_.clear_feasible_solution_callback(); }
  lns_population_feed_t(const lns_population_feed_t&)            = delete;
  lns_population_feed_t& operator=(const lns_population_feed_t&) = delete;

  bool best_feasible(std::vector<f_t>& assignment)
  {
    return best_.adopt(std::numeric_limits<f_t>::infinity(), assignment);
  }

 private:
  population_t<i_t, f_t>& population_;
  fj_cpu_shared_incumbent_t<i_t, f_t> best_;
};

}  // namespace cuopt::mathematical_optimization::mip
