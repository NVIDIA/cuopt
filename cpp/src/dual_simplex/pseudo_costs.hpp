/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/branch_and_bound_worker.hpp>
#include <dual_simplex/logger.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/omp_helpers.hpp>
#include <utilities/pcgenerator.hpp>

#include <omp.h>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct reliability_branching_settings_t {
  // Lower bound for the maximum number of LP iterations for a single trial branching
  i_t lower_max_lp_iter = 10;

  // Upper bound for the maximum number of LP iterations for a single trial branching
  i_t upper_max_lp_iter = 500;

  // Priority of the tasks created when running the trial branching in parallel.
  // Set to 1 to have the same priority as the other tasks.
  i_t task_priority = 5;

  // The maximum number of candidates initialized by strong branching in a single
  // node
  i_t max_num_candidates = 100;

  // Define the maximum number of iteration spent in strong branching.
  // Let `bnb_lp_iter` = total number of iterations in B&B, then
  // `max iter in strong branching = bnb_lp_factor * bnb_lp_iter + bnb_lp_offset`.
  // This is used for determining the `reliable_threshold`.
  f_t bnb_lp_factor = 0.5;
  i_t bnb_lp_offset = 100000;

  // Maximum and minimum points in curve to determine the value
  // of the `reliable_threshold` based on the current number of LP
  // iterations in strong branching and B&B. Since it is a
  // a curve, the actual value of `reliable_threshold` may be
  // higher than `max_reliable_threshold`.
  // Only used when `reliable_threshold` is negative
  i_t max_reliable_threshold = 5;
  i_t min_reliable_threshold = 1;
};

template <typename i_t, typename f_t>
class pseudo_costs_t {
 public:
  explicit pseudo_costs_t(i_t num_variables)
    : pseudo_cost_sum_down(num_variables),
      pseudo_cost_sum_up(num_variables),
      pseudo_cost_num_down(num_variables),
      pseudo_cost_num_up(num_variables),
      pseudo_cost_mutex_up(num_variables),
      pseudo_cost_mutex_down(num_variables)
  {
  }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node_ptr, f_t leaf_objective);

  void resize(i_t num_variables)
  {
    pseudo_cost_sum_down.assign(num_variables, 0);
    pseudo_cost_sum_up.assign(num_variables, 0);
    pseudo_cost_num_down.assign(num_variables, 0);
    pseudo_cost_num_up.assign(num_variables, 0);
    pseudo_cost_mutex_up.resize(num_variables);
    pseudo_cost_mutex_down.resize(num_variables);
  }

  void initialized(i_t& num_initialized_down,
                   i_t& num_initialized_up,
                   f_t& pseudo_cost_down_avg,
                   f_t& pseudo_cost_up_avg) const;

  f_t obj_estimate(const std::vector<i_t>& fractional,
                   const std::vector<f_t>& solution,
                   f_t lower_bound,
                   logger_t& log);

  i_t variable_selection(const std::vector<i_t>& fractional,
                         const std::vector<f_t>& solution,
                         logger_t& log);

  i_t reliable_variable_selection(mip_node_t<i_t, f_t>* node_ptr,
                                  const std::vector<i_t>& fractional,
                                  const std::vector<f_t>& solution,
                                  const simplex_solver_settings_t<i_t, f_t>& settings,
                                  const std::vector<variable_type_t>& var_types,
                                  branch_and_bound_worker_t<i_t, f_t>* worker,
                                  const branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
                                  f_t upper_bound,
                                  int max_num_tasks,
                                  logger_t& log);

  void update_pseudo_costs_from_strong_branching(const std::vector<i_t>& fractional,
                                                 const std::vector<f_t>& root_soln);

  f_t calculate_pseudocost_score(i_t j,
                                 const std::vector<f_t>& solution,
                                 f_t pseudo_cost_up_avg,
                                 f_t pseudo_cost_down_avg) const;

  reliability_branching_settings_t<i_t, f_t> reliability_branching_settings;

  std::vector<omp_atomic_t<f_t>> pseudo_cost_sum_up;
  std::vector<omp_atomic_t<f_t>> pseudo_cost_sum_down;
  std::vector<omp_atomic_t<i_t>> pseudo_cost_num_up;
  std::vector<omp_atomic_t<i_t>> pseudo_cost_num_down;
  std::vector<f_t> strong_branch_down;
  std::vector<f_t> strong_branch_up;
  std::vector<omp_mutex_t> pseudo_cost_mutex_up;
  std::vector<omp_mutex_t> pseudo_cost_mutex_down;
  omp_atomic_t<i_t> num_strong_branches_completed = 0;
  omp_atomic_t<int64_t> strong_branching_lp_iter  = 0;
};

template <typename i_t, typename f_t>
void strong_branching(const user_problem_t<i_t, f_t>& original_problem,
                      const lp_problem_t<i_t, f_t>& original_lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      f_t start_time,
                      const std::vector<variable_type_t>& var_types,
                      const std::vector<f_t> root_soln,
                      const std::vector<i_t>& fractional,
                      f_t root_obj,
                      const std::vector<variable_status_t>& root_vstatus,
                      const std::vector<f_t>& edge_norms,
                      pseudo_costs_t<i_t, f_t>& pc);

}  // namespace cuopt::linear_programming::dual_simplex
