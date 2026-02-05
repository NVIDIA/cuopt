/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/diving_heuristics.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/node_queue.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/macros.cuh>
#include <utilities/omp_helpers.hpp>

#include <omp.h>
#include <functional>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

enum class mip_status_t {
  OPTIMAL    = 0,  // The optimal integer solution was found
  UNBOUNDED  = 1,  // The problem is unbounded
  INFEASIBLE = 2,  // The problem is infeasible
  TIME_LIMIT = 3,  // The solver reached a time limit
  NODE_LIMIT = 4,  // The maximum number of nodes was reached (not implemented)
  NUMERICAL  = 5,  // The solver encountered a numerical error
  UNSET      = 6,  // The status is not set
};

// Indicate the search and variable selection algorithms used by each thread
// in B&B (See [1]).
//
// [1] T. Achterberg, “Constraint Integer Programming,” PhD, Technischen Universität Berlin,
// Berlin, 2007. doi: 10.14279/depositonce-1634.
enum class bnb_worker_type_t {
  BEST_FIRST         = 0,  // Best-First + Plunging.
  PSEUDOCOST_DIVING  = 1,  // Pseudocost diving (9.2.5)
  LINE_SEARCH_DIVING = 2,  // Line search diving (9.2.4)
  GUIDED_DIVING = 3,  // Guided diving (9.2.3). If no incumbent is found yet, use pseudocost diving.
  COEFFICIENT_DIVING = 4  // Coefficient diving (9.2.1)
};

template <typename i_t, typename f_t>
class bounds_strengthening_t;

template <typename i_t, typename f_t>
void upper_bound_callback(f_t upper_bound);

template <typename i_t, typename f_t>
struct bnb_stats_t {
  f_t start_time                        = 0.0;
  omp_atomic_t<f_t> total_lp_solve_time = 0.0;
  omp_atomic_t<i_t> nodes_explored      = 0;
  omp_atomic_t<i_t> nodes_unexplored    = 0;
  omp_atomic_t<f_t> total_lp_iters      = 0;

  // This should only be used by the main thread
  omp_atomic_t<f_t> last_log             = 0.0;
  omp_atomic_t<i_t> nodes_since_last_log = 0;
};

template <typename i_t, typename f_t>
class branch_and_bound_t {
 public:
  branch_and_bound_t(const user_problem_t<i_t, f_t>& user_problem,
                     const simplex_solver_settings_t<i_t, f_t>& solver_settings);

  // Set an initial guess based on the user_problem. This should be called before solve.
  void set_initial_guess(const std::vector<f_t>& user_guess) { guess_ = user_guess; }

  // Set the root solution found by PDLP
  void set_root_relaxation_solution(const std::vector<f_t>& primal,
                                    const std::vector<f_t>& dual,
                                    const std::vector<f_t>& reduced_costs,
                                    f_t objective,
                                    f_t user_objective,
                                    i_t iterations)
  {
    if (!is_root_solution_set) {
      root_crossover_soln_.x              = primal;
      root_crossover_soln_.y              = dual;
      root_crossover_soln_.z              = reduced_costs;
      root_objective_                     = objective;
      root_crossover_soln_.objective      = objective;
      root_crossover_soln_.user_objective = user_objective;
      root_crossover_soln_.iterations     = iterations;
      root_crossover_solution_set_.store(true, std::memory_order_release);
    }
  }

  // Set a solution based on the user problem during the course of the solve
  void set_new_solution(const std::vector<f_t>& solution);

  void set_user_bound_callback(std::function<void(f_t)> callback)
  {
    user_bound_callback_ = std::move(callback);
  }

  void set_concurrent_lp_root_solve(bool enable) { enable_concurrent_lp_root_solve_ = enable; }

  // Repair a low-quality solution from the heuristics.
  bool repair_solution(const std::vector<f_t>& leaf_edge_norms,
                       const std::vector<f_t>& potential_solution,
                       f_t& repaired_obj,
                       std::vector<f_t>& repaired_solution) const;

  f_t get_lower_bound();
  bool enable_concurrent_lp_root_solve() const { return enable_concurrent_lp_root_solve_; }
  std::atomic<int>* get_root_concurrent_halt() { return &root_concurrent_halt_; }
  void set_root_concurrent_halt(int value) { root_concurrent_halt_ = value; }
  lp_status_t solve_root_relaxation(simplex_solver_settings_t<i_t, f_t> const& lp_settings);

  // The main entry routine. Returns the solver status and populates solution with the incumbent.
  mip_status_t solve(mip_solution_t<i_t, f_t>& solution);

 private:
  const user_problem_t<i_t, f_t>& original_problem_;
  const simplex_solver_settings_t<i_t, f_t> settings_;

  // Initial guess.
  std::vector<f_t> guess_;

  // LP relaxation
  csr_matrix_t<i_t, f_t> Arow_;
  lp_problem_t<i_t, f_t> original_lp_;
  std::vector<i_t> new_slacks_;
  std::vector<variable_type_t> var_types_;

  // Variable locks (see definition 3.3 from T. Achterberg, “Constraint Integer Programming,”
  // PhD, Technischen Universität Berlin, Berlin, 2007. doi: 10.14279/depositonce-1634).
  // Here we assume that the constraints are in the form `Ax = b, l <= x <= u`.
  std::vector<i_t> var_up_locks_;
  std::vector<i_t> var_down_locks_;

  // Local lower bounds for each thread
  std::vector<omp_atomic_t<f_t>> local_lower_bounds_;

  // Mutex for upper bound
  omp_mutex_t mutex_upper_;

  // Global variable for upper bound
  omp_atomic_t<f_t> upper_bound_;

  // Global variable for incumbent. The incumbent should be updated with the upper bound
  mip_solution_t<i_t, f_t> incumbent_;

  // Structure with the general info of the solver.
  bnb_stats_t<i_t, f_t> exploration_stats_;

  // Mutex for repair
  omp_mutex_t mutex_repair_;
  std::vector<std::vector<f_t>> repair_queue_;

  // Variables for the root node in the search tree.
  std::vector<variable_status_t> root_vstatus_;
  std::vector<variable_status_t> crossover_vstatus_;
  f_t root_objective_;
  lp_solution_t<i_t, f_t> root_relax_soln_;
  lp_solution_t<i_t, f_t> root_crossover_soln_;
  std::vector<f_t> edge_norms_;
  std::atomic<bool> root_crossover_solution_set_{false};
  bool enable_concurrent_lp_root_solve_{false};
  std::atomic<int> root_concurrent_halt_{0};
  bool is_root_solution_set{false};

  // Pseudocosts
  pseudo_costs_t<i_t, f_t> pc_;

  // Heap storing the nodes waiting to be explored.
  node_queue_t<i_t, f_t> node_queue_;

  // Search tree
  search_tree_t<i_t, f_t> search_tree_;

  // Count the number of subtrees that are currently being explored.
  omp_atomic_t<i_t> active_subtrees_;

  // Global status of the solver.
  omp_atomic_t<mip_status_t> solver_status_;
  omp_atomic_t<bool> is_running{false};

  omp_atomic_t<bool> should_report_;

  // In case, a best-first thread encounters a numerical issue when solving a node,
  // its blocks the progression of the lower bound.
  omp_atomic_t<f_t> lower_bound_ceiling_;
  std::function<void(f_t)> user_bound_callback_;

  void report_heuristic(f_t obj);
  void report(char symbol, f_t obj, f_t lower_bound, i_t node_depth);
  void update_user_bound(f_t lower_bound);

  // Set the final solution.
  void set_final_solution(mip_solution_t<i_t, f_t>& solution, f_t lower_bound);

  // Update the incumbent solution with the new feasible solution
  // found during branch and bound.
  void add_feasible_solution(f_t leaf_objective,
                             const std::vector<f_t>& leaf_solution,
                             i_t leaf_depth,
                             bnb_worker_type_t thread_type);

  // Repairs low-quality solutions from the heuristics, if it is applicable.
  void repair_heuristic_solutions();

  // Ramp-up phase of the solver, where we greedily expand the tree until
  // there is enough unexplored nodes. This is done recursively using OpenMP tasks.
  void exploration_ramp_up(mip_node_t<i_t, f_t>* node, i_t initial_heap_size);

  // We use best-first to pick the `start_node` and then perform a depth-first search
  // from this node (i.e., a plunge). It can only backtrack to a sibling node.
  // Unexplored nodes in the subtree are inserted back into the global heap.
  void plunge_from(i_t task_id,
                   mip_node_t<i_t, f_t>* start_node,
                   lp_problem_t<i_t, f_t>& leaf_problem,
                   bounds_strengthening_t<i_t, f_t>& node_presolver,
                   basis_update_mpf_t<i_t, f_t>& basis_update,
                   std::vector<i_t>& basic_list,
                   std::vector<i_t>& nonbasic_list);

  // Each "main" thread pops a node from the global heap and then performs a plunge
  // (i.e., a shallow dive) into the subtree determined by the node.
  void best_first_thread(i_t task_id);

  // Perform a deep dive in the subtree determined by the `start_node` in order
  // to find integer feasible solutions.
  void dive_from(mip_node_t<i_t, f_t>& start_node,
                 const std::vector<f_t>& start_lower,
                 const std::vector<f_t>& start_upper,
                 lp_problem_t<i_t, f_t>& leaf_problem,
                 bounds_strengthening_t<i_t, f_t>& node_presolver,
                 basis_update_mpf_t<i_t, f_t>& basis_update,
                 std::vector<i_t>& basic_list,
                 std::vector<i_t>& nonbasic_list,
                 bnb_worker_type_t diving_type);

  // Each diving thread pops the first node from the dive queue and then performs
  // a deep dive into the subtree determined by the node.
  void diving_thread(bnb_worker_type_t diving_type);

  // Solve the LP relaxation of a leaf node
  dual::status_t solve_node_lp(mip_node_t<i_t, f_t>* node_ptr,
                               lp_problem_t<i_t, f_t>& leaf_problem,
                               lp_solution_t<i_t, f_t>& leaf_solution,
                               basis_update_mpf_t<i_t, f_t>& basis_factors,
                               std::vector<i_t>& basic_list,
                               std::vector<i_t>& nonbasic_list,
                               bounds_strengthening_t<i_t, f_t>& node_presolver,
                               bnb_worker_type_t thread_type,
                               bool recompute_bounds_and_basis,
                               const std::vector<f_t>& root_lower,
                               const std::vector<f_t>& root_upper,
                               bnb_stats_t<i_t, f_t>& stats,
                               logger_t& log);

  // Update the tree based on the LP relaxation. Returns the status
  // of the node and, if appropriated, the preferred rounding direction
  // when visiting the children.
  std::pair<node_status_t, rounding_direction_t> update_tree(mip_node_t<i_t, f_t>* node_ptr,
                                                             search_tree_t<i_t, f_t>& search_tree,
                                                             lp_problem_t<i_t, f_t>& leaf_problem,
                                                             lp_solution_t<i_t, f_t>& leaf_solution,
                                                             bnb_worker_type_t thread_type,
                                                             dual::status_t lp_status,
                                                             logger_t& log);

  // Selects the variable to branch on.
  branch_variable_t<i_t> variable_selection(mip_node_t<i_t, f_t>* node_ptr,
                                            const std::vector<i_t>& fractional,
                                            const std::vector<f_t>& solution,
                                            bnb_worker_type_t type);
};

}  // namespace cuopt::linear_programming::dual_simplex
