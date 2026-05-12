/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver.cuh>

#include <utilities/omp_helpers.hpp>

#include <vector>

namespace cuopt::linear_programming::detail {

// forward declare
template <typename i_t, typename f_t>
class diversity_manager_t;

struct rins_settings_t {
  int node_freq                     = 100;
  int nodes_after_later_improvement = 200;
  double min_fixrate                = 0.3;
  double max_fixrate                = 0.8;
  double min_fractional_ratio       = 0.3;
  double min_time_limit             = 3.;
  double target_mip_gap             = 0.03;
  bool objective_cut                = true;
};

template <typename i_t, typename f_t>
class rins_t;

template <typename i_t, typename f_t>
class rins_t {
 public:
  rins_t(mip_solver_context_t<i_t, f_t>& context,
         diversity_manager_t<i_t, f_t>& dm,
         rins_settings_t settings = rins_settings_t());

  void node_callback(const std::vector<f_t>& solution, f_t objective);
  void new_best_incumbent_callback(const std::vector<f_t>& solution);
  void enable();

  void run_rins();

  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t>* problem_ptr;
  diversity_manager_t<i_t, f_t>& dm;
  rins_settings_t settings;

  // need a separate handle for RINS to operate on a separate stream and prevent graph capture
  // issues
  std::unique_ptr<problem_t<i_t, f_t>> problem_copy;
  raft::handle_t rins_handle;

  std::vector<f_t> lp_optimal_solution;

  f_t fixrate{0.5};
  i_t total_calls{0};
  i_t total_success{0};
  f_t time_limit{10.};
  i_t seed;

  omp_atomic_t<bool> enabled{false};
  omp_atomic_t<f_t> lower_bound{0.};

  omp_atomic_t<i_t> node_count{0};
  omp_atomic_t<i_t> node_count_at_last_rins{0};
  omp_atomic_t<i_t> node_count_at_last_improvement{0};
  omp_atomic_t<bool> launch_new_task{true};
};

}  // namespace cuopt::linear_programming::detail
