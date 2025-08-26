/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/types.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>
#include "cuopt/linear_programming/mip/solver_settings.hpp"
#include "dual_simplex/mip_node.hpp"

namespace cuopt::linear_programming::dual_simplex {

enum class mip_status_t {
  OPTIMAL    = 0,
  UNBOUNDED  = 1,
  INFEASIBLE = 2,
  TIME_LIMIT = 3,
  NODE_LIMIT = 4,
  NUMERICAL  = 5,
  UNSET      = 6
};

template <typename i_t, typename f_t>
void upper_bound_callback(f_t upper_bound);

template <typename i_t, typename f_t>
class branch_and_bound_t {
 public:
  branch_and_bound_t(const user_problem_t<i_t, f_t>& user_problem,
                     const simplex_solver_settings_t<i_t, f_t>& solver_settings);

  // Set an initial guess based on the user_problem. This should be called before solve.
  void set_initial_guess(const std::vector<f_t>& user_guess) { guess = user_guess; }

  // Set a solution based on the user problem during the course of the solve
  void set_new_solution(const std::vector<f_t>& solution);

  bool repair_solution(const std::vector<variable_status_t>& root_vstatus,
                       const std::vector<f_t>& leaf_edge_norms,
                       const std::vector<f_t>& potential_solution,
                       f_t& repaired_obj,
                       std::vector<f_t>& repaired_solution) const;

  f_t get_upper_bound();

  f_t get_lower_bound();

  // The main entry routine. Returns the solver status and populates solution with the incumbent.
  mip_status_t solve(mip_solution_t<i_t, f_t>& solution);

 private:
  const user_problem_t<i_t, f_t>& original_problem;
  const simplex_solver_settings_t<i_t, f_t> settings;

  std::vector<f_t> guess;

  lp_problem_t<i_t, f_t> original_lp;
  std::vector<i_t> new_slacks;
  std::vector<variable_type_t> var_types;
  // Mutex for lower bound
  std::mutex mutex_lower;
  // Global variable for lower bound
  f_t lower_bound_;

  // Mutex for upper bound
  std::mutex mutex_upper;
  // Global variable for upper bound
  f_t upper_bound_;
  // Global variable for incumbent. The incumbent should be updated with the upper bound
  mip_solution_t<i_t, f_t> incumbent_;

  // Mutex for gap
  std::mutex mutex_gap;
  // Global variable for gap
  f_t gap_;

  // Mutex for branching
  std::mutex mutex_branching;
  bool currently_branching_;

  // Global variable for stats
  std::mutex mutex_stats;

  // Note  that floating point atomics are only supported in C++20
  struct stats_t {
    f_t start_time                    = 0.0;
    f_t total_lp_solve_time           = 0.0;
    std::atomic<i_t> nodes_explored   = 0;
    std::atomic<i_t> unexplored_nodes = 0;
    f_t total_lp_iters                = 0;
    std::atomic<i_t> num_nodes        = 0;
  } stats_;

  // Mutex for repair
  std::mutex mutex_repair;
  std::vector<std::vector<f_t>> repair_queue;

  // Pseudocosts
  std::mutex mutex_pseudocosts;
  pseudo_costs_t<i_t, f_t> pc;

  // Search tree
  std::unique_ptr<mip_node_t<i_t, f_t>> root_node_;
  std::vector<variable_status_t> root_vstatus;
  std::vector<f_t> edge_norms;

  void repair_heuristic_solutions(const f_t& lower_bound, mip_solution_t<i_t, f_t>& solution);

  void report(f_t last_log,
              i_t nodes_explored,
              i_t nodes_unexplored,
              f_t gap,
              f_t upper_bound,
              f_t lower_bound,
              i_t leaf_depth);

  mip_status_t explore_tree(mip_node_t<i_t, f_t>* start_node,
                            f_t root_objective,
                            mip_solution_t<i_t, f_t>& solution);

  mip_status_t dive(f_t root_objective,
                    i_t branch_var,
                    f_t branch_var_val,
                    std::vector<variable_status_t>& root_vstatus,
                    mip_solution_t<i_t, f_t>& solution,
                    bool enable_reporting);

  void branch(mip_node_t<i_t, f_t>* parent_node,
              i_t branch_var,
              f_t branch_var_val,
              const std::vector<variable_status_t>& parent_vstatus);

  mip_status_t solve_root_relaxation(f_t& root_objective,
                                     lp_solution_t<i_t, f_t>& root_relax_soln,
                                     std::vector<variable_status_t>& root_vstatus);

  mip_status_t solve_leaf_lp(mip_node_t<i_t, f_t>* node_ptr,
                             lp_problem_t<i_t, f_t>& leaf_problem,
                             f_t upper_bound,
                             f_t lower_bound,
                             i_t nodes_explored,
                             i_t unexplored_nodes);
};

}  // namespace cuopt::linear_programming::dual_simplex
