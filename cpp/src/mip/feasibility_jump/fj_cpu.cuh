/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <functional>
#include <unordered_set>
#include <vector>

#include <mip/feasibility_jump/feasibility_jump.cuh>

namespace cuopt::linear_programming::detail {

// NOTE: this seems an easy pick for reflection/xmacros once this is available (C++26?)
// Maintaining a single source of truth for all members would be nice
template <typename i_t, typename f_t>
struct fj_cpu_climber_t {
  fj_cpu_climber_t()                                                             = default;
  fj_cpu_climber_t(const fj_cpu_climber_t<i_t, f_t>& other)                      = delete;
  fj_cpu_climber_t<i_t, f_t>& operator=(const fj_cpu_climber_t<i_t, f_t>& other) = delete;

  fj_cpu_climber_t(fj_cpu_climber_t<i_t, f_t>&& other)                      = default;
  fj_cpu_climber_t<i_t, f_t>& operator=(fj_cpu_climber_t<i_t, f_t>&& other) = default;

  problem_t<i_t, f_t>* pb_ptr;
  fj_settings_t settings;
  typename fj_t<i_t, f_t>::climber_data_t::view_t view;
  // Host copies of device data as struct members
  std::vector<f_t> h_reverse_coefficients;
  std::vector<i_t> h_reverse_constraints;
  std::vector<i_t> h_reverse_offsets;
  std::vector<f_t> h_coefficients;
  std::vector<i_t> h_offsets;
  std::vector<i_t> h_variables;
  std::vector<f_t> h_obj_coeffs;
  std::vector<typename type_2<f_t>::type> h_var_bounds;
  std::vector<f_t> h_cstr_lb;
  std::vector<f_t> h_cstr_ub;
  std::vector<var_t> h_var_types;
  std::vector<i_t> h_is_binary_variable;
  std::vector<i_t> h_objective_vars;
  std::vector<i_t> h_binary_indices;

  std::vector<i_t> h_tabu_nodec_until;
  std::vector<i_t> h_tabu_noinc_until;
  std::vector<i_t> h_tabu_lastdec;
  std::vector<i_t> h_tabu_lastinc;

  std::vector<f_t> h_lhs;
  std::vector<f_t> h_lhs_sumcomp;
  std::vector<f_t> h_cstr_left_weights;
  std::vector<f_t> h_cstr_right_weights;
  f_t max_weight;
  std::vector<f_t> h_assignment;
  std::vector<f_t> h_best_assignment;
  f_t h_objective_weight;
  f_t h_incumbent_objective;
  f_t h_best_objective;
  i_t last_feasible_entrance_iter{0};
  i_t iterations;
  std::unordered_set<i_t> violated_constraints;
  std::unordered_set<i_t> satisfied_constraints;
  bool feasible_found{false};
  bool trigger_early_lhs_recomputation{false};
  f_t total_violations{0};

  // Timing data structures
  std::vector<double> find_lift_move_times;
  std::vector<double> find_mtm_move_viol_times;
  std::vector<double> find_mtm_move_sat_times;
  std::vector<double> apply_move_times;
  std::vector<double> update_weights_times;
  std::vector<double> compute_score_times;

  i_t hit_count{0};
  i_t miss_count{0};

  i_t candidate_move_hits[3]   = {0};
  i_t candidate_move_misses[3] = {0};

  // vector<bool> is actually likely beneficial here since we're memory bound
  std::vector<bool> flip_move_computed;
  ;
  // CSR nnz offset -> (delta, score)
  std::vector<std::pair<f_t, fj_staged_score_t>> cached_mtm_moves;

  // CSC (transposed!) nnz-offset-indexed constraint bounds (lb, ub)
  // std::pair<f_t, f_t> better compile down to 16 bytes!! GCC do your job!
  std::vector<std::pair<f_t, f_t>> cached_cstr_bounds;

  std::vector<bool> var_bitmap;
  std::vector<i_t> iter_mtm_vars;

  i_t mtm_viol_samples{25};
  i_t mtm_sat_samples{15};
  i_t nnz_samples{50000};
  i_t perturb_interval{100};

  i_t log_interval{1000};
  i_t diversity_callback_interval{3000};
  i_t timing_stats_interval{5000};

  std::function<void(f_t, const std::vector<f_t>&)> improvement_callback{nullptr};
  std::function<void(f_t, const std::vector<f_t>&)> diversity_callback{nullptr};
  std::string log_prefix{""};

  std::atomic<bool> halted{false};
};

}  // namespace cuopt::linear_programming::detail
