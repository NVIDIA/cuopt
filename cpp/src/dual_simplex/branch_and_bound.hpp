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
  void set_initial_guess(const std::vector<f_t>& user_guess) { guess_ = user_guess; }

  // Set a solution based on the user problem during the course of the solve
  void set_new_solution(const std::vector<f_t>& solution);

  // Repair a low-quality solution from the heuristics.
  bool repair_solution(const std::vector<f_t>& leaf_edge_norms,
                       const std::vector<f_t>& potential_solution,
                       f_t& repaired_obj,
                       std::vector<f_t>& repaired_solution) const;

  f_t get_upper_bound();

  f_t get_lower_bound();

  // The main entry routine. Returns the solver status and populates solution with the incumbent.
  mip_status_t solve(mip_solution_t<i_t, f_t>& solution);

 private:
  const user_problem_t<i_t, f_t>& original_problem_;
  const simplex_solver_settings_t<i_t, f_t> settings_;

  // Initial guess.
  std::vector<f_t> guess_;

  lp_problem_t<i_t, f_t> original_lp_;
  std::vector<i_t> new_slacks_;
  std::vector<variable_type_t> var_types_;

  // Mutex for lower bound
  std::mutex mutex_lower_;

  // Global variable for lower bound
  f_t lower_bound_;

  // Mutex for upper bound
  std::mutex mutex_upper_;

  // Global variable for upper bound
  f_t upper_bound_;

  // Global variable for incumbent. The incumbent should be updated with the upper bound
  mip_solution_t<i_t, f_t> incumbent_;

  // Mutex for gap
  std::mutex mutex_gap_;

  // Global variable for gap
  f_t gap_;

  // Mutex for branching
  std::mutex mutex_branching_;
  bool currently_branching_;

  // Global variable for stats
  std::mutex mutex_stats_;

  // Note that floating point atomics are only supported in C++20.
  struct stats_t {
    f_t start_time                    = 0.0;
    f_t total_lp_solve_time           = 0.0;
    std::atomic<i_t> nodes_explored   = 0;
    std::atomic<i_t> nodes_unexplored = 0;
    f_t total_lp_iters                = 0;
    std::atomic<i_t> num_nodes        = 0;
  } stats_;

  // Mutex for repair
  std::mutex mutex_repair_;
  std::vector<std::vector<f_t>> repair_queue_;

  // Variables for the root node in the search tree.
  std::vector<variable_status_t> root_vstatus_;
  f_t root_objective_;
  lp_solution_t<i_t, f_t> root_relax_soln_;
  std::vector<f_t> edge_norms_;

  // Pseudocosts
  pseudo_costs_t<i_t, f_t> pc_;
  std::mutex mutex_pc_;

  // Update the status of the nodes in the search tree.
  void update_tree(mip_node_t<i_t, f_t>* node_ptr, node_status_t status);

  // Update the incumbent solution with the new feasible solution.
  // found during branch and bound.
  void add_feasible_solution(f_t leaf_objective,
                             const std::vector<f_t>& leaf_solution,
                             i_t leaf_depth,
                             char symbol);

  // Repairs low-quality solutions from the heuristics, if it is applicable.
  void repair_heuristic_solutions();

  // Explore the search tree using the best-first search strategy.
  mip_status_t explore_tree(i_t branch_var, mip_solution_t<i_t, f_t>& solution);

  // Explore the search tree using the depth-first search strategy.
  mip_status_t dive(i_t branch_var, mip_solution_t<i_t, f_t>& solution);

  // Branch the current node, creating two children.
  void branch(mip_node_t<i_t, f_t>* parent_node,
              i_t branch_var,
              f_t branch_var_val,
              const std::vector<variable_status_t>& parent_vstatus);

  // Solve the LP relaxation of a leaf node.
  mip_status_t solve_node_lp(mip_node_t<i_t, f_t>* node_ptr,
                             lp_problem_t<i_t, f_t>& leaf_problem,
                             csc_matrix_t<i_t, f_t>& Arow,
                             const std::vector<variable_type_t>& var_types,
                             f_t upper_bound);

  // Solve the LP relaxation of a leaf node using the dual simplex method.
  dual::status_t node_dual_simplex(i_t leaf_id,
                                   lp_problem_t<i_t, f_t>& leaf_problem,
                                   std::vector<variable_status_t>& leaf_vstatus,
                                   lp_solution_t<i_t, f_t>& leaf_solution,
                                   std::vector<bool>& bounds_changed,
                                   csc_matrix_t<i_t, f_t>& Arow,
                                   f_t upper_bound,
                                   logger_t& log);
};

}  // namespace cuopt::linear_programming::dual_simplex
