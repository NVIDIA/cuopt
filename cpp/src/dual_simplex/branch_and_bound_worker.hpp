/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/mip_node.hpp>
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

  bool recompute_basis  = true;
  bool recompute_bounds = true;

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
    start_node    = &internal_node;

    start_lower     = original_lp.lower;
    start_upper     = original_lp.upper;
    search_strategy = type;
    lower_bound     = node->lower_bound;
    is_active       = true;

    std::fill(bounds_changed.begin(), bounds_changed.end(), false);
    node->get_variable_bounds(start_lower, start_upper, bounds_changed);
    return node_presolver.bounds_strengthening(settings, bounds_changed, start_lower, start_upper);
  }

  // Set the variables bounds for the LP relaxation of the current node.
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

template <typename i_t, typename f_t>
class branch_and_bound_worker_pool_t {
 public:
  void init(i_t num_workers,
            const lp_problem_t<i_t, f_t>& original_lp,
            const csr_matrix_t<i_t, f_t>& Arow,
            const std::vector<variable_type_t>& var_type,
            const simplex_solver_settings_t<i_t, f_t>& settings)
  {
    workers_.resize(num_workers);
    num_idle_workers_ = num_workers;
    for (i_t i = 0; i < num_workers; ++i) {
      workers_[i] = std::make_unique<branch_and_bound_worker_t<i_t, f_t>>(
        i, original_lp, Arow, var_type, settings);
      idle_workers_.push_front(i);
    }

    is_initialized = true;
  }

  // Here, we are assuming that the scheduler is the only
  // thread that can retrieve/pop an idle worker.
  branch_and_bound_worker_t<i_t, f_t>* get_idle_worker()
  {
    std::lock_guard<omp_mutex_t> lock(mutex_);
    if (idle_workers_.empty()) {
      return nullptr;
    } else {
      i_t idx = idle_workers_.front();
      return workers_[idx].get();
    }
  }

  // Here, we are assuming that the scheduler is the only
  // thread that can retrieve/pop an idle worker.
  void pop_idle_worker()
  {
    std::lock_guard<omp_mutex_t> lock(mutex_);
    if (!idle_workers_.empty()) {
      idle_workers_.pop_front();
      num_idle_workers_--;
    }
  }

  void return_worker_to_pool(branch_and_bound_worker_t<i_t, f_t>* worker)
  {
    worker->is_active = false;
    std::lock_guard<omp_mutex_t> lock(mutex_);
    idle_workers_.push_back(worker->worker_id);
    num_idle_workers_++;
  }

  f_t get_lower_bound()
  {
    f_t lower_bound = std::numeric_limits<f_t>::infinity();

    if (is_initialized) {
      for (i_t i = 0; i < workers_.size(); ++i) {
        if (workers_[i]->search_strategy == BEST_FIRST && workers_[i]->is_active) {
          lower_bound = std::min(workers_[i]->lower_bound.load(), lower_bound);
        }
      }
    }

    return lower_bound;
  }

  i_t num_idle_workers() { return num_idle_workers_; }

 private:
  // Worker pool
  std::vector<std::unique_ptr<branch_and_bound_worker_t<i_t, f_t>>> workers_;
  bool is_initialized = false;

  omp_mutex_t mutex_;
  std::deque<i_t> idle_workers_;
  omp_atomic_t<i_t> num_idle_workers_;
};

template <typename f_t, typename i_t>
std::vector<search_strategy_t> get_search_strategies(
  diving_heuristics_settings_t<i_t, f_t> settings)
{
  std::vector<search_strategy_t> types;
  types.reserve(num_search_strategies);
  types.push_back(BEST_FIRST);
  if (settings.pseudocost_diving != 0) { types.push_back(PSEUDOCOST_DIVING); }
  if (settings.line_search_diving != 0) { types.push_back(LINE_SEARCH_DIVING); }
  if (settings.guided_diving != 0) { types.push_back(GUIDED_DIVING); }
  if (settings.coefficient_diving != 0) { types.push_back(COEFFICIENT_DIVING); }
  return types;
}

template <typename i_t>
std::array<i_t, num_search_strategies> get_max_workers(
  i_t num_workers, const std::vector<search_strategy_t>& strategies)
{
  std::array<i_t, num_search_strategies> max_num_workers;
  max_num_workers.fill(0);

  i_t bfs_workers             = std::max(strategies.size() == 1 ? num_workers : num_workers / 4, 1);
  max_num_workers[BEST_FIRST] = bfs_workers;

  i_t diving_workers = (num_workers - bfs_workers);
  i_t m              = strategies.size() - 1;

  for (size_t i = 1, k = 0; i < strategies.size(); ++i) {
    i_t start                      = (double)k * diving_workers / m;
    i_t end                        = (double)(k + 1) * diving_workers / m;
    max_num_workers[strategies[i]] = end - start;
    ++k;
  }

  return max_num_workers;
}

}  // namespace cuopt::linear_programming::dual_simplex
