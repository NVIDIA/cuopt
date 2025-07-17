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

#include <mip/local_search/rounding/bounds_repair.cuh>
#include <mip/presolve/bounds_presolve.cuh>
#include <mip/presolve/conditional_bound_strengthening.cuh>
#include <mip/presolve/multi_probe.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <raft/random/rng_device.cuh>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

struct repair_stats_t {
  size_t repair_attempts                           = 0;
  size_t repair_success                            = 0;
  size_t intermediate_repair_success               = 0;
  size_t total_repair_loops                        = 0;
  double total_time_spent_on_repair                = 0.;
  double total_time_spent_bounds_prop_after_repair = 0.;
  double total_time_spent_on_bounds_prop           = 0.;
};

template <typename i_t, typename f_t>
struct probing_config_t {
  probing_config_t(i_t n_vars, const raft::handle_t* handle_ptr) : probing_values(n_vars) {}
  bool use_balanced_probing  = false;
  i_t n_of_fixed_from_first  = 0;
  i_t n_of_fixed_from_second = 0;
  std::vector<thrust::pair<f_t, f_t>> probing_values;
};

template <typename i_t, typename f_t>
struct constraint_prop_t {
  constraint_prop_t(mip_solver_context_t<i_t, f_t>& context);
  bool apply_round(solution_t<i_t, f_t>& sol,
                   f_t lp_run_time_after_feasible,
                   timer_t& timer,
                   std::optional<std::reference_wrapper<probing_config_t<i_t, f_t>>>
                     probing_config = std::nullopt);
  void sort_by_implied_slack_consumption(solution_t<i_t, f_t>& sol,
                                         raft::device_span<i_t> vars,
                                         bool problem_ii);
  void sort_by_interval_and_frac(solution_t<i_t, f_t>& sol,
                                 raft::device_span<i_t> vars,
                                 std::mt19937 rng);

  bool find_integer(solution_t<i_t, f_t>& sol,
                    solution_t<i_t, f_t>& orig_sol,
                    f_t lp_run_time_after_feasible,
                    timer_t& timer,
                    std::optional<std::reference_wrapper<probing_config_t<i_t, f_t>>>
                      probing_config = std::nullopt);
  void find_set_integer_vars(solution_t<i_t, f_t>& sol, rmm::device_uvector<i_t>& set_vars);
  void find_unset_integer_vars(solution_t<i_t, f_t>& sol, rmm::device_uvector<i_t>& set_vars);
  thrust::pair<f_t, f_t> generate_double_probing_pair(
    const solution_t<i_t, f_t>& sol,
    const solution_t<i_t, f_t>& orig_sol,
    i_t unset_var_idx,
    const std::optional<std::reference_wrapper<probing_config_t<i_t, f_t>>> probing_config,
    bool bulk_rounding);
  std::tuple<f_t, f_t, f_t> probing_values(const solution_t<i_t, f_t>& sol,
                                           const solution_t<i_t, f_t>& orig_sol,
                                           i_t idx);
  void update_host_assignment(const solution_t<i_t, f_t>& sol);
  void set_host_bounds(const solution_t<i_t, f_t>& sol);
  bool probe(solution_t<i_t, f_t>& sol,
             problem_t<i_t, f_t>* original_problem,
             const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals,
             size_t* set_count_ptr,
             rmm::device_uvector<i_t>& unset_vars,
             std::optional<std::reference_wrapper<probing_config_t<i_t, f_t>>> probing_config);
  void collapse_crossing_bounds(problem_t<i_t, f_t>& problem,
                                problem_t<i_t, f_t>& orig_problem,
                                const raft::handle_t* handle_ptr);
  void set_bounds_on_fixed_vars(solution_t<i_t, f_t>& sol);
  void round_close_values(solution_t<i_t, f_t>& sol, rmm::device_uvector<i_t>& unset_integer_vars);
  void sort_by_frac(solution_t<i_t, f_t>& sol, raft::device_span<i_t> vars);
  void restore_bounds(solution_t<i_t, f_t>& sol);
  void save_bounds(solution_t<i_t, f_t>& sol);
  void copy_bounds(rmm::device_uvector<f_t>& output_lb,
                   rmm::device_uvector<f_t>& output_ub,
                   const rmm::device_uvector<f_t>& input_lb,
                   const rmm::device_uvector<f_t>& input_ub,
                   const raft::handle_t* handle_ptr);
  void copy_bounds(rmm::device_uvector<f_t>& output_lb,
                   rmm::device_uvector<f_t>& output_ub,
                   rmm::device_uvector<f_t>& output_assignment,
                   const rmm::device_uvector<f_t>& input_lb,
                   const rmm::device_uvector<f_t>& input_ub,
                   const rmm::device_uvector<f_t>& input_assignment,
                   const raft::handle_t* handle_ptr);
  void restore_original_bounds(solution_t<i_t, f_t>& sol, solution_t<i_t, f_t>& orig_sol);
  std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>> generate_bulk_rounding_vector(
    const solution_t<i_t, f_t>& sol,
    const solution_t<i_t, f_t>& orig_sol,
    const std::vector<i_t>& host_vars_to_set,
    const std::optional<std::reference_wrapper<probing_config_t<i_t, f_t>>> probing_config);
  bool is_problem_ii(problem_t<i_t, f_t>& problem);
  void restore_original_bounds_on_unfixed(problem_t<i_t, f_t>& problem,
                                          problem_t<i_t, f_t>& original_problem,
                                          const raft::handle_t* handle_ptr);
  bool run_repair_procedure(problem_t<i_t, f_t>& problem,
                            problem_t<i_t, f_t>& original_problem,
                            timer_t& timer,
                            const raft::handle_t* handle_ptr);
  bool handle_fixed_vars(
    solution_t<i_t, f_t>& sol,
    problem_t<i_t, f_t>* original_problem,
    const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals,
    size_t* set_count_ptr,
    rmm::device_uvector<i_t>& unset_vars);
  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t> temp_problem;
  solution_t<i_t, f_t> temp_sol;
  bound_presolve_t<i_t, f_t> bounds_update;
  multi_probe_t<i_t, f_t> multi_probe;
  bounds_repair_t<i_t, f_t> bounds_repair;
  rmm::device_uvector<i_t> set_vars;
  rmm::device_uvector<i_t> unset_vars;
  conditional_bound_strengthening_t<i_t, f_t> conditional_bounds_update;
  rmm::device_uvector<f_t> lb_restore;
  rmm::device_uvector<f_t> ub_restore;
  rmm::device_uvector<f_t> assignment_restore;
  std::vector<f_t> curr_host_assignment;
  raft::random::PCGenerator rng;
  bool recovery_mode                 = false;
  bool rounding_ii                   = false;
  i_t selected_update                = 0;
  i_t bounds_prop_interval           = 1;
  i_t n_iter_in_recovery             = 0;
  i_t max_n_failed_repair_iterations = 1;
  timer_t max_timer{0.};
  bool use_probing_cache = true;
  static repair_stats_t repair_stats;
  bool single_rounding_only = false;
};

}  // namespace cuopt::linear_programming::detail
