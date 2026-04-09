/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/constants.hpp>
#include <branch_and_bound/mip_node.hpp>

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bounds_strengthening.hpp>

#include <utilities/pcgenerator.hpp>

#include <deque>
#include <vector>

#include "node_queue.hpp"

namespace cuopt::linear_programming::dual_simplex {

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
  using float_type = f_t;
  using int_type   = i_t;

  i_t worker_id;
  omp_atomic_t<search_strategy_t> search_strategy;
  omp_atomic_t<bool> is_active;
  omp_atomic_t<f_t> lower_bound;
  omp_atomic_t<i_t> node_depth;
  omp_atomic_t<i_t> integer_infeasible;

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

  pcgenerator_t rng;

  bool recompute_basis  = true;
  bool recompute_bounds = true;

  branch_and_bound_worker_t(i_t worker_id,
                            const lp_problem_t<i_t, f_t>& original_lp,
                            const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<variable_type_t>& var_type,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            uint64_t rng_offset = 0)
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
      rng(settings.random_seed + pcgenerator_t::default_seed + rng_offset + worker_id,
          pcgenerator_t::default_stream ^ (worker_id + rng_offset))
  {
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
};

template <typename i_t, typename f_t>
class bfs_worker_t : public branch_and_bound_worker_t<i_t, f_t> {
 public:
  using Base = branch_and_bound_worker_t<i_t, f_t>;
  bfs_worker_t(i_t worker_id,
               const lp_problem_t<i_t, f_t>& original_lp,
               const csr_matrix_t<i_t, f_t>& Arow,
               const std::vector<variable_type_t>& var_type,
               const simplex_solver_settings_t<i_t, f_t>& settings,
               uint64_t rng_offset = 0)
    : Base(worker_id, original_lp, Arow, var_type, settings, rng_offset)
  {
    Base::start_lower     = original_lp.lower;
    Base::start_upper     = original_lp.upper;
    Base::search_strategy = BEST_FIRST;
  }

  f_t get_lower_bound()
  {
    f_t lower_bound = std::numeric_limits<f_t>::infinity();

    if (Base::is_active) {
      lower_bound = std::min(node_queue.get_lower_bound(), Base::lower_bound.load());
    }

    return lower_bound;
  }

  node_queue_t<i_t, f_t> node_queue;
};

template <typename i_t, typename f_t>
class diving_worker_t : public branch_and_bound_worker_t<i_t, f_t> {
 public:
  using Base = branch_and_bound_worker_t<i_t, f_t>;
  using Base::Base;

  // Initialize the worker for diving, setting the `start_node`, `start_lower` and
  // `start_upper`. Returns `true` if the starting node is feasible via
  // bounds propagation.
  bool init(mip_node_t<i_t, f_t>* node,
            search_strategy_t type,
            const lp_problem_t<i_t, f_t>& original_lp,
            const simplex_solver_settings_t<i_t, f_t>& settings)
  {
    start_node            = node->detach_copy();
    Base::start_lower     = original_lp.lower;
    Base::start_upper     = original_lp.upper;
    Base::search_strategy = type;
    Base::lower_bound     = node->lower_bound;
    Base::is_active       = true;

    std::fill(Base::bounds_changed.begin(), Base::bounds_changed.end(), false);
    node->get_variable_bounds(Base::start_lower, Base::start_upper, Base::bounds_changed);
    return Base::node_presolver.bounds_strengthening(
      settings, Base::bounds_changed, Base::start_lower, Base::start_upper);
  }

  f_t get_lower_bound()
  {
    return Base::is_active ? Base::lower_bound.load() : std::numeric_limits<f_t>::infinity();
  }

  mip_node_t<i_t, f_t> start_node;
};

}  // namespace cuopt::linear_programming::dual_simplex
