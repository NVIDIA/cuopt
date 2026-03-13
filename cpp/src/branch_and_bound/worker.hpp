/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/mip_node.hpp>

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/phase2.hpp>

#include <utilities/pcgenerator.hpp>

#include <array>
#include <deque>
#include <mutex>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

constexpr int num_search_strategies = 5;

// Indicate the search and variable selection algorithms used by each thread
// in B&B (See [1]).
//
// [1] T. Achterberg, “Constraint Integer Programming,” PhD, Technischen Universität Berlin,
// Berlin, 2007. doi: 10.14279/depositonce-1634.
enum search_strategy_t : int {
  BEST_FIRST         = 0,  // Best-First + Plunging.
  PSEUDOCOST_DIVING  = 1,  // Pseudocost diving (9.2.5)
  LINE_SEARCH_DIVING = 2,  // Line search diving (9.2.4)
  GUIDED_DIVING      = 3,  // Guided diving (9.2.3).
  COEFFICIENT_DIVING = 4   // Coefficient diving (9.2.1)
};

template <typename i_t, typename f_t>
struct branch_and_bound_stats_t {
  f_t start_time                         = 0.0;
  omp_atomic_t<f_t> total_lp_solve_time  = 0.0;
  omp_atomic_t<int64_t> nodes_explored   = 0;
  omp_atomic_t<int64_t> nodes_unexplored = 0;
  omp_atomic_t<int64_t> total_lp_iters   = 0;
  omp_atomic_t<i_t> nodes_since_last_log = 0;
  omp_atomic_t<f_t> last_log             = 0.0;
};

template <typename i_t, typename f_t>
class branch_and_bound_worker_t {
 public:
  const i_t worker_id;
  omp_atomic_t<search_strategy_t> search_strategy;
  omp_atomic_t<bool> is_active;
  omp_atomic_t<f_t> lower_bound;

  lp_problem_t<i_t, f_t> leaf_problem;
  lp_solution_t<i_t, f_t> leaf_solution;
  std::vector<f_t> leaf_edge_norms;

  basis_update_mpf_t<i_t, f_t> basis_factors;
  std::vector<i_t> basic_list;
  std::vector<i_t> nonbasic_list;

  bounds_strengthening_t<i_t, f_t> node_presolver;
  std::vector<bool> bounds_changed;

  std::vector<f_t> start_lower;
  std::vector<f_t> start_upper;
  mip_node_t<i_t, f_t>* start_node;

  pcgenerator_t rng;

  bool recompute_basis                    = true;
  bool recompute_bounds                   = true;

  branch_and_bound_worker_t(i_t worker_id,
                            const lp_problem_t<i_t, f_t>& original_lp,
                            const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<variable_type_t>& var_type,
                            const simplex_solver_settings_t<i_t, f_t>& settings)
    : worker_id(worker_id),
      search_strategy(BEST_FIRST),
      is_active(false),
      lower_bound(-std::numeric_limits<f_t>::infinity()),
      leaf_problem(original_lp),
      leaf_solution(original_lp.num_rows, original_lp.num_cols),
      basis_factors(original_lp.num_rows, settings.refactor_frequency),
      basic_list(original_lp.num_rows),
      nonbasic_list(),
      node_presolver(leaf_problem, Arow, {}, var_type),
      bounds_changed(original_lp.num_cols, false),
      rng(settings.random_seed + pcgenerator_t::default_seed + worker_id,
          pcgenerator_t::default_stream ^ worker_id)
  {
  }

  // Set the `start_node` for best-first search.
  void init_best_first(mip_node_t<i_t, f_t>* node, const lp_problem_t<i_t, f_t>& original_lp)
  {
    start_node      = node;
    start_lower     = original_lp.lower;
    start_upper     = original_lp.upper;
    search_strategy = BEST_FIRST;
    lower_bound     = node->lower_bound;
    is_active       = true;
  }

  // Initialize the worker for diving, setting the `start_node`, `start_lower` and
  // `start_upper`. Returns `true` if the starting node is feasible via
  // bounds propagation.
  bool init_diving(mip_node_t<i_t, f_t>* node,
                   search_strategy_t type,
                   const lp_problem_t<i_t, f_t>& original_lp,
                   const simplex_solver_settings_t<i_t, f_t>& settings)
  {
    internal_node = node->detach_copy();
    start_node      = &internal_node;

    start_lower     = original_lp.lower;
    start_upper     = original_lp.upper;
    search_strategy = type;
    lower_bound     = node->lower_bound;
    is_active       = true;

    std::fill(bounds_changed.begin(), bounds_changed.end(), false);
    node->get_variable_bounds(start_lower, start_upper, bounds_changed);
    return node_presolver.bounds_strengthening(settings, bounds_changed, start_lower, start_upper);
  }

  // Set the variables bounds for the LP relaxation in the current node.
  bool set_lp_variable_bounds(mip_node_t<i_t, f_t>* node_ptr,
                              const simplex_solver_settings_t<i_t, f_t>& settings)
  {
    // Reset the bound_changed markers
    std::fill(bounds_changed.begin(), bounds_changed.end(), false);

    // Set the correct bounds for the leaf problem
    if (recompute_bounds) {
      leaf_problem.lower = start_lower;
      leaf_problem.upper = start_upper;
      node_ptr->get_variable_bounds(leaf_problem.lower, leaf_problem.upper, bounds_changed);

    } else {
      node_ptr->update_branched_variable_bounds(
        leaf_problem.lower, leaf_problem.upper, bounds_changed);
    }

    return node_presolver.bounds_strengthening(
      settings, bounds_changed, leaf_problem.lower, leaf_problem.upper);
  }

 private:
  // For diving, we need to store the full node instead of
  // of just a pointer, since it is not stored in the tree anymore.
  // To keep the same interface across all worker types,
  // this will be used as a temporary storage and
  // will be pointed by `start_node`.
  // For exploration, this will not be used.
  mip_node_t<i_t, f_t> internal_node;
};

}  // namespace cuopt::linear_programming::dual_simplex
