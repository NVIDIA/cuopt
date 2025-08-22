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

#include <mutex>
#include <string>
#include <vector>
#include "cuopt/linear_programming/mip/solver_settings.hpp"

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
  struct stats_t {
    f_t lower_bound         = 0.0;
    f_t upper_bound         = inf;
    f_t gap                 = 0.0;
    f_t total_lp_solve_time = 0.0;
    i_t nodes_explored      = 0;
    i_t unexplored_nodes    = 0;
    f_t total_lp_iters      = 0;
    i_t num_nodes           = 0;
    mip_status_t status     = mip_status_t::UNSET;
  };

  branch_and_bound_t(const user_problem_t<i_t, f_t>& user_problem,
                     const simplex_solver_settings_t<i_t, f_t>& solver_settings,
                     const search_settings_t strategy);

  // Set an initial guess based on the user_problem. This should be called before solve.
  void set_initial_guess(const std::vector<f_t>& user_guess) { guess = user_guess; }

  // Set a solution based on the user problem during the course of the solve
  void set_new_solution(const std::vector<f_t>& solution);

  bool repair_solution(const std::vector<variable_status_t>& root_vstatus,
                       const std::vector<f_t>& leaf_edge_norms,
                       const std::vector<f_t>& potential_solution,
                       f_t& repaired_obj,
                       std::vector<f_t>& repaired_solution) const;

  // The main entry routine. Returns the solver status and populates solution with the incumbent.
  mip_status_t solve(mip_solution_t<i_t, f_t>& solution);

 private:
  const user_problem_t<i_t, f_t>& original_problem;
  const simplex_solver_settings_t<i_t, f_t> settings;
  search_settings_t search_settings;

  f_t start_time;
  std::vector<f_t> guess;

  lp_problem_t<i_t, f_t> original_lp;
  std::vector<i_t> new_slacks;
  std::vector<variable_type_t> var_types;

  void repair_heuristic_solutions(const std::vector<variable_status_t>& root_vstatus,
                                  const std::vector<f_t>& edge_norms,
                                  const f_t& lower_bound,
                                  mip_solution_t<i_t, f_t>& incumbent,
                                  mip_solution_t<i_t, f_t>& solution);

  void best_first_solve(stats_t& stats,
                        f_t root_objective,
                        i_t branch_var,
                        f_t branch_var_val,
                        std::vector<variable_status_t>& root_vstatus,
                        std::vector<f_t>& edge_norms,
                        pseudo_costs_t<i_t, f_t> pc,
                        mip_solution_t<i_t, f_t>& incumbent,
                        mip_solution_t<i_t, f_t>& solution);

  void depth_first_solve(stats_t& stats,
                         f_t root_objective,
                         i_t branch_var,
                         f_t branch_var_val,
                         std::vector<variable_status_t>& root_vstatus,
                         std::vector<f_t>& edge_norms,
                         pseudo_costs_t<i_t, f_t> pc,
                         mip_solution_t<i_t, f_t>& incumbent,
                         mip_solution_t<i_t, f_t>& solution,
                         bool enable_reporting);

  void branch(mip_node_t<i_t, f_t>* parent_node,
              i_t branch_var,
              f_t branch_var_val,
              std::vector<variable_status_t>& parent_vstatus,
              stats_t& stats);

  void add_feasible_solution(mip_node_t<i_t, f_t>* leaf_ptr,
                             f_t leaf_objective,
                             const std::vector<f_t>& leaf_sol,
                             mip_solution_t<i_t, f_t>& incumbent,
                             stats_t& stats,
                             char symbol);

  mip_status_t solve_root_relaxation(f_t& root_objective,
                                     lp_solution_t<i_t, f_t>& root_relax_soln,
                                     std::vector<variable_status_t>& root_vstatus,
                                     std::vector<f_t>& edge_norms,
                                     stats_t& stats);

  dual::status_t solve_leaf_lp(mip_node_t<i_t, f_t>* node_ptr,
                               lp_problem_t<i_t, f_t>& leaf_problem,
                               std::vector<variable_status_t>& leaf_vstatus,
                               lp_solution_t<i_t, f_t>& leaf_solution,
                               std::vector<f_t>& edge_norms,
                               stats_t& stats);
};

}  // namespace cuopt::linear_programming::dual_simplex
