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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/types.hpp>

#include <omp.h>
#include <queue>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

enum class mip_status_t {
  OPTIMAL    = 0,
  UNBOUNDED  = 1,
  INFEASIBLE = 2,
  TIME_LIMIT = 3,
  NODE_LIMIT = 4,
  NUMERICAL  = 5,
  UNSET      = 6,
  RUNNING    = 7,
  FINISHED   = 8
};

// Since we are using OpenMP, we omp_lock_t instead of std::mutex.
class omp_mutex_t {
 public:
  omp_mutex_t() { omp_init_lock(&lock_); }
  ~omp_mutex_t() { omp_destroy_lock(&lock_); }
  void lock() { omp_set_lock(&lock_); }
  void unlock() { omp_unset_lock(&lock_); }

 private:
  omp_lock_t lock_;
};

template <typename i_t, typename f_t>
void upper_bound_callback(f_t upper_bound);

// Note that floating point atomics are only supported in C++20. So, we
// are using omp atomic operations instead.
template <typename i_t, typename f_t>
class branch_and_bound_t {
 public:
  using heap_t = std::priority_queue<mip_node_t<i_t, f_t>*,
                                     std::vector<mip_node_t<i_t, f_t>*>,
                                     node_compare_t<i_t, f_t>>;

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
  mip_status_t get_status();
  i_t get_heap_size();

  // The main entry routine. Returns the solver status and populates solution with the incumbent.
  mip_status_t solve(mip_solution_t<i_t, f_t>& solution);

 private:
  const user_problem_t<i_t, f_t>& original_problem_;
  const simplex_solver_settings_t<i_t, f_t> settings_;

  // Initial guess.
  std::vector<f_t> guess_;

  // LP relaxation
  lp_problem_t<i_t, f_t> original_lp_;
  std::vector<i_t> new_slacks_;
  std::vector<variable_type_t> var_types_;

  std::vector<f_t> lower_bounds_;

  // Mutex for upper bound
  omp_mutex_t mutex_upper_;

  // Global variable for upper bound
  f_t upper_bound_;

  // Global variable for incumbent. The incumbent should be updated with the upper bound
  mip_solution_t<i_t, f_t> incumbent_;

  // Structure with the general info of the solver.
  struct stats_t {
    f_t start_time          = 0.0;
    f_t total_lp_solve_time = 0.0;
    i_t nodes_explored      = 0;
    i_t nodes_unexplored    = 0;
    f_t total_lp_iters      = 0;
    i_t num_nodes           = 0;
  } stats_;

  // Mutex for repair
  omp_mutex_t mutex_repair_;
  std::vector<std::vector<f_t>> repair_queue_;

  // Variables for the root node in the search tree.
  std::vector<variable_status_t> root_vstatus_;
  f_t root_objective_;
  lp_solution_t<i_t, f_t> root_relax_soln_;
  std::vector<f_t> edge_norms_;

  // Pseudocosts
  pseudo_costs_t<i_t, f_t> pc_;
  omp_mutex_t mutex_pc_;

  // Search tree
  omp_mutex_t mutex_search_tree_;
  std::unique_ptr<mip_node_t<i_t, f_t>> search_tree_;

  // Heap storing the nodes to be explored.
  omp_mutex_t mutex_heap_;
  heap_t heap_;

  // Global status
  mip_status_t status_;

  i_t node_depth_threshold_;

  // Update the status of the nodes in the search tree.
  void update_tree(mip_node_t<i_t, f_t>* node_ptr, node_status_t status);

  // Repairs low-quality solutions from the heuristics, if it is applicable.
  void repair_heuristic_solutions();

  // Explore the search tree using the best-first search strategy.
  void explore_subtree(mip_node_t<i_t, f_t>* start_node);

  // Branch the current node, creating two children.
  void branch(mip_node_t<i_t, f_t>* parent_node,
              i_t branch_var,
              f_t branch_var_val,
              const std::vector<variable_status_t>& parent_vstatus);

  // Solve the LP relaxation of a leaf node.
  mip_status_t solve_node_lp(mip_node_t<i_t, f_t>* node_ptr,
                             lp_problem_t<i_t, f_t>& leaf_problem,
                             csc_matrix_t<i_t, f_t>& Arow,
                             f_t upper_bound);

  // Solve the LP relaxation of a leaf node using the dual simplex method.
  dual::status_t node_dual_simplex(i_t leaf_id,
                                   lp_problem_t<i_t, f_t>& leaf_problem,
                                   std::vector<variable_status_t>& leaf_vstatus,
                                   lp_solution_t<i_t, f_t>& leaf_solution,
                                   std::vector<bool>& bounds_changed,
                                   csc_matrix_t<i_t, f_t>& Arow,
                                   f_t upper_bound);
};

}  // namespace cuopt::linear_programming::dual_simplex
