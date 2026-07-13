/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/constants.hpp>
#include <branch_and_bound/diving_heuristics.hpp>
#include <branch_and_bound/mip_node.hpp>
#include <branch_and_bound/node_queue.hpp>
#include <branch_and_bound/symmetry.hpp>

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bounds_strengthening.hpp>

#include <utilities/pcgenerator.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct branch_and_bound_stats_t {
  f_t start_time                         = 0.0;
  omp_atomic_t<f_t> total_lp_solve_time  = 0.0;
  omp_atomic_t<int64_t> nodes_explored   = 0;
  omp_atomic_t<int64_t> nodes_unexplored = 0;
  // Tracks the number of nodes being solved by the workers at a given time
  omp_atomic_t<i_t> nodes_being_solved = 0;

  omp_atomic_t<int64_t> total_lp_iters   = 0;
  omp_atomic_t<i_t> nodes_since_last_log = 0;
  omp_atomic_t<f_t> last_log             = 0.0;

  omp_atomic_t<int64_t> orbital_fixing_nodes              = 0;
  omp_atomic_t<int64_t> orbital_fixings_applied           = 0;
  omp_atomic_t<int64_t> orbital_conflict_nodes            = 0;
  omp_atomic_t<int64_t> lexical_reduction_nodes           = 0;
  omp_atomic_t<int64_t> lexical_reduction_fixings_applied = 0;
  omp_atomic_t<int64_t> lexical_reduction_pruned_nodes    = 0;
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

  simplex::lp_problem_t<i_t, f_t> leaf_problem;
  simplex::lp_solution_t<i_t, f_t> leaf_solution;
  std::vector<simplex::variable_status_t> leaf_vstatus;
  std::vector<f_t> leaf_edge_norms;

  simplex::basis_update_mpf_t<i_t, f_t> basis_factors;
  std::vector<i_t> basic_list;
  std::vector<i_t> nonbasic_list;

  simplex::bounds_strengthening_t<i_t, f_t> node_presolver;
  std::vector<bool> bounds_changed;

  std::vector<f_t> start_lower;
  std::vector<f_t> start_upper;

  pcgenerator_t rng;

  std::unique_ptr<orbital_fixing_t<i_t, f_t>> orbital_fixing;
  std::unique_ptr<lexical_reduction_t<i_t, f_t>> lexical_reduction;
  mip_symmetry_t<i_t, f_t>* symmetry_ptr = nullptr;

  void ensure_orbital_fixing()
  {
    if (orbital_fixing == nullptr && symmetry_ptr != nullptr) {
      orbital_fixing = std::make_unique<orbital_fixing_t<i_t, f_t>>(*symmetry_ptr);
    }
    if (lexical_reduction == nullptr && symmetry_ptr != nullptr) {
      lexical_reduction =
        std::make_unique<lexical_reduction_t<i_t, f_t>>(symmetry_ptr->num_original_vars);
    }
  }

  bool recompute_basis  = true;
  bool recompute_bounds = true;

  branch_and_bound_worker_t(i_t worker_id,
                            const simplex::lp_problem_t<i_t, f_t>& original_lp,
                            const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<simplex::variable_type_t>& var_type,
                            const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                            uint64_t rng_offset = 0)
    : worker_id(worker_id),
      search_strategy(BEST_FIRST),
      is_active(false),
      lower_bound(-std::numeric_limits<f_t>::infinity()),
      leaf_problem(original_lp),
      leaf_solution(original_lp.num_rows, original_lp.num_cols),
      leaf_vstatus(original_lp.num_cols),
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
                              const simplex::simplex_solver_settings_t<i_t, f_t>& settings)
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

  void set_active() { is_active = true; }
};

template <typename i_t, typename f_t>
class bfs_worker_t : public branch_and_bound_worker_t<i_t, f_t> {
 public:
  using Base = branch_and_bound_worker_t<i_t, f_t>;
  bfs_worker_t(i_t worker_id,
               const simplex::lp_problem_t<i_t, f_t>& original_lp,
               const csr_matrix_t<i_t, f_t>& Arow,
               const std::vector<simplex::variable_type_t>& var_type,
               const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
               uint64_t rng_offset = 0)
    : Base(worker_id, original_lp, Arow, var_type, settings, rng_offset)
  {
    this->start_lower     = original_lp.lower;
    this->start_upper     = original_lp.upper;
    this->search_strategy = BEST_FIRST;

    max_diving_workers.fill(0);
    active_diving_workers.fill(0);
    total_active_diving_workers = 0;
  }

  void set_inactive() { this->is_active = false; }

  // Steal nodes from another worker
  bool steal_from(bfs_worker_t* victim, i_t nodes_to_steal)
  {
    if (!victim || nodes_to_steal < 1) return false;
    if (victim == this || !victim->is_active ||
        victim->node_queue.best_first_queue_size() < 2 * nodes_to_steal) {
      return false;
    }

    return node_queue.steal_from(victim->node_queue, nodes_to_steal);
  }

  // Calculate the number of diving workers that this worker can launch. Having a fixed number
  // of workers allows the solver to be more deterministic.
  void calculate_num_diving_workers(i_t num_bfs_workers,
                                    i_t total_diving_workers,
                                    const mip_diving_hyper_params_t<i_t, f_t>& settings)
  {
    i_t num_active = 0;
    for (i_t i = 1; i < num_search_strategies; ++i) {
      num_active += is_search_strategy_enabled(search_strategies[i], settings);
    }

    total_max_diving_workers = 0;
    max_diving_workers.fill(0);
    if (num_active == 0) { return; }

    for (size_t i = 1, k = 0; i < num_search_strategies; ++i) {
      if (is_search_strategy_enabled(search_strategies[i], settings)) {
        // Calculate the number of workers for a given diving heuristic
        auto [type_start, type_end] = calculate_index_range(k, total_diving_workers, num_active);
        i_t workers_per_type        = type_end - type_start;

        // Calculate the number of diving workers allocated to this (best-first) worker
        auto [start, end] =
          calculate_index_range(this->worker_id, workers_per_type, num_bfs_workers);
        max_diving_workers[i] = end - start;
        total_max_diving_workers += max_diving_workers[i];
        ++k;
      }
    }
  }

  // The worker-local node heap.
  node_queue_t<i_t, f_t> node_queue;

  // The number of diving workers of each type that this (best-first) worker can launch.
  std::array<i_t, num_search_strategies> max_diving_workers;

  // The number of active diving workers of each type associated with this (best-first) worker.
  std::array<omp_atomic_t<i_t>, num_search_strategies> active_diving_workers;

  // Keep track of the total number of active diving worker that are associated with this
  // (best-first) worker
  omp_atomic_t<i_t> total_active_diving_workers{0};

  // The maximum number of diving worker that are associated with this
  // (best-first) worker
  i_t total_max_diving_workers{0};
};

template <typename i_t, typename f_t>
class diving_worker_t : public branch_and_bound_worker_t<i_t, f_t> {
 public:
  using Base = branch_and_bound_worker_t<i_t, f_t>;
  using Base::Base;

  // Apply bound strengthening to the starting variable bounds
  bool presolve_start_bounds(const simplex::simplex_solver_settings_t<i_t, f_t>& settings)
  {
    return this->node_presolver.bounds_strengthening(
      settings, this->bounds_changed, this->start_lower, this->start_upper);
  }

  // Set this node inactive
  void set_inactive()
  {
    if (!this->is_active.load()) { return; }
    assert(bfs_worker != nullptr);
    assert(bfs_worker->active_diving_workers[this->search_strategy].load() > 0);
    assert(bfs_worker->total_active_diving_workers.load() > 0);

    this->is_active = false;
    --bfs_worker->active_diving_workers[this->search_strategy];
    --bfs_worker->total_active_diving_workers;
  }

  f_t get_lower_bound() { return this->lower_bound; }

  mip_node_t<i_t, f_t> start_node;

  // The best-first worker that is associated with this diving worker. Used for controlling the
  // number of active diving workers.
  bfs_worker_t<i_t, f_t>* bfs_worker{nullptr};
};

}  // namespace cuopt::mathematical_optimization::mip
