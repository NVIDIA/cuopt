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

#include <mip/presolve/load_balanced_bounds_presolve.cuh>
#include <mip/problem/load_balanced_problem.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <raft/random/rng_device.cuh>
#include <utilities/timer.hpp>
#include "lb_bounds_repair.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct lb_constraint_prop_t {
  lb_constraint_prop_t(mip_solver_context_t<i_t, f_t>& context);
  bool apply_round(
    solution_t<i_t, f_t>& sol,
    f_t lp_run_time_after_feasible,
    timer_t& timer,
    std::optional<std::vector<thrust::pair<f_t, f_t>>> probing_candidates = std::nullopt);
  void sort_by_implied_slack_consumption(
    problem_t<i_t, f_t>& original_problem,
    load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
    raft::device_span<i_t> vars,
    bool problem_ii);
  void sort_by_interval_and_frac(load_balanced_problem_t<i_t, f_t>& problem,
                                 rmm::device_uvector<f_t>& assignment,
                                 raft::device_span<i_t> vars,
                                 std::mt19937 rng);

  bool find_integer(rmm::device_uvector<f_t>& assignment,
                    load_balanced_problem_t<i_t, f_t>& problem,
                    load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
                    solution_t<i_t, f_t>& orig_sol,
                    f_t lp_run_time_after_feasible,
                    timer_t& timer,
                    std::optional<std::vector<thrust::pair<f_t, f_t>>> probing_candidates);
  std::tuple<f_t, f_t, f_t> probing_values(
    load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
    const solution_t<i_t, f_t>& orig_sol,
    i_t idx);
  void update_host_assignment(const rmm::device_uvector<f_t>& assignment,
                              const raft::handle_t* handle_ptr);
  bool probe(rmm::device_uvector<f_t>& assignment,
             load_balanced_problem_t<i_t, f_t>* problem,
             load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
             problem_t<i_t, f_t>* original_problem,
             const std::vector<thrust::pair<i_t, f_t>>& var_probe_val_pairs,
             size_t* set_count_ptr,
             rmm::device_uvector<i_t>& unset_vars);

  void collapse_crossing_bounds(load_balanced_problem_t<i_t, f_t>* problem,
                                problem_t<i_t, f_t>& orig_problem,
                                const raft::handle_t* handle_ptr);
  void restore_bounds(load_balanced_problem_t<i_t, f_t>* problem,
                      rmm::device_uvector<f_t>* assignment,
                      const raft::handle_t* handle_ptr);
  void save_bounds(load_balanced_problem_t<i_t, f_t>& problem,
                   rmm::device_uvector<f_t>& assignment,
                   const raft::handle_t* handle_ptr);
  void copy_bounds(rmm::device_uvector<f_t>& output_variable_bounds,
                   rmm::device_uvector<f_t>& output_assignment,
                   const rmm::device_uvector<f_t>& input_variable_bounds,
                   const rmm::device_uvector<f_t>& input_assignment,
                   const raft::handle_t* handle_ptr);
  std::vector<thrust::pair<i_t, f_t>> generate_bulk_rounding_vector(
    load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
    const solution_t<i_t, f_t>& orig_sol,
    const std::vector<i_t>& host_vars_to_set,
    const std::optional<std::vector<thrust::pair<f_t, f_t>>> probing_candidates);
  bool is_problem_ii(load_balanced_problem_t<i_t, f_t>& problem,
                     load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update);
  void restore_original_bounds_on_unfixed(load_balanced_problem_t<i_t, f_t>* problem,
                                          problem_t<i_t, f_t>& original_problem,
                                          const raft::handle_t* handle_ptr);
  bool run_repair_procedure(load_balanced_problem_t<i_t, f_t>* problem,
                            load_balanced_bounds_presolve_t<i_t, f_t>& lb_bounds_update,
                            problem_t<i_t, f_t>& original_problem,
                            timer_t& timer,
                            const raft::handle_t* handle_ptr);

  mip_solver_context_t<i_t, f_t>& context;
  load_balanced_problem_t<i_t, f_t> temp_problem;
  load_balanced_bounds_presolve_t<i_t, f_t> bounds_update;
  lb_bounds_repair_t<i_t, f_t> bounds_repair;
  rmm::device_uvector<i_t> unset_vars;
  rmm::device_uvector<f_t> temp_assignment;
  rmm::device_uvector<f_t> bounds_restore;
  rmm::device_uvector<f_t> assignment_restore;
  std::vector<f_t> curr_host_assignment;
  raft::random::PCGenerator rng;
  bool recovery_mode       = false;
  bool rounding_ii         = false;
  i_t bounds_prop_interval = 1;
  i_t n_iter_in_recovery   = 0;
  timer_t max_timer{0.};
  bool use_probing_cache = true;

  size_t repair_attempts                           = 0;
  size_t repair_success                            = 0;
  size_t intermediate_repair_success               = 0;
  size_t total_repair_loops                        = 0;
  double total_time_spent_on_repair                = 0.;
  double total_time_spent_bounds_prop_after_repair = 0.;
  double total_time_spent_on_bounds_prop           = 0.;
};

}  // namespace cuopt::linear_programming::detail
