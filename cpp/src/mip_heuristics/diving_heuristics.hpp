/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/mip/diving_hyper_params.hpp>

#include <branch_and_bound/pseudo_costs.hpp>
#include <branch_and_bound/worker.hpp>
#include <branch_and_bound/worker_pool.hpp>

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/domain.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class diving_worker_t : public branch_and_bound_worker_t<i_t, f_t> {
 public:
  using Base = branch_and_bound_worker_t<i_t, f_t>;

  diving_worker_t(i_t worker_id,
                  const simplex::lp_problem_t<i_t, f_t>& original_lp,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  const std::vector<simplex::variable_type_t>& var_type,
                  const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                  pseudo_costs_t<i_t, f_t>& pc,
                  const std::vector<f_t>& root_solution,
                  const std::vector<f_t>& root_edge_norm,
                  const std::vector<i_t>& new_slacks,
                  branch_and_bound_t<i_t, f_t>* branch_and_bound_ptr,
                  splitmix64_t& seed_gen)
    : Base(worker_id,
           original_lp,
           Arow,
           var_type,
           settings,
           pc,
           root_solution,
           root_edge_norm,
           new_slacks,
           branch_and_bound_ptr,
           seed_gen)
  {
    this->search_strategy = search_strategy_t::COEFFICIENT_DIVING;
  }

  // Set this node inactive
  void cleanup()
  {
    if (!this->is_active.load(std::memory_order::relaxed)) { return; }
    this->is_active = false;
    this->domain.clear();

    if (bfs_worker) {
      assert(bfs_worker->active_diving_workers.load(std::memory_order::relaxed) > 0);
      --bfs_worker->active_diving_workers;
    }
  }

  void dive(const simplex::lp_problem_t<i_t, f_t>& original_lp,
            const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
            const branch_and_bound_stats_t<i_t, f_t>& solver_stats);

  // Classify the node after solving its LP relaxation. HAS_CHILDREN means the node is fractional
  // and within the cutoff; the caller is responsible for branching.
  node_status_t classify_node(mip_node_t<i_t, f_t>* node_ptr,
                              simplex::dual_status_t lp_status,
                              f_t upper_bound,
                              const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                              std::vector<i_t>& fractional,
                              f_t& leaf_obj);

  branch_variable_t<i_t> select_branch_variable(
    const std::vector<i_t>& fractional,
    const simplex::simplex_solver_settings_t<i_t, f_t>& settings);

  mip_node_t<i_t, f_t> start_node;

  // The best-first worker that is associated with this diving worker. Used for controlling the
  // number of active diving workers.
  bfs_worker_t<i_t, f_t>* bfs_worker{nullptr};
};

template <typename i_t, typename f_t>
using diving_worker_pool_t = worker_pool_t<diving_worker_t<i_t, f_t>>;

template <typename i_t, typename f_t>
branch_variable_t<i_t> line_search_diving(const std::vector<i_t>& fractional,
                                          const std::vector<f_t>& solution,
                                          const std::vector<f_t>& root_solution,
                                          simplex::logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> pseudocost_diving(pseudo_costs_t<i_t, f_t>& pc,
                                         const std::vector<i_t>& fractional,
                                         const std::vector<f_t>& solution,
                                         const std::vector<f_t>& root_solution,
                                         simplex::logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> guided_diving(pseudo_costs_t<i_t, f_t>& pc,
                                     const std::vector<i_t>& fractional,
                                     const std::vector<f_t>& solution,
                                     const std::vector<f_t>& incumbent,
                                     simplex::logger_t& log);

// Calculate the variable locks assuming that the constraints
// has the following format: `Ax = b`.
template <typename i_t, typename f_t>
void calculate_variable_locks(const simplex::lp_problem_t<i_t, f_t>& lp_problem,
                              std::vector<i_t>& up_locks,
                              std::vector<i_t>& down_locks);

template <typename i_t, typename f_t>
branch_variable_t<i_t> coefficient_diving(const simplex::lp_problem_t<i_t, f_t>& lp_problem,
                                          const std::vector<i_t>& fractional,
                                          const std::vector<f_t>& solution,
                                          const std::vector<i_t>& up_locks,
                                          const std::vector<i_t>& down_locks,
                                          simplex::logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> farkas_diving(const simplex::lp_problem_t<i_t, f_t>& lp,
                                     const std::vector<i_t>& fractional,
                                     const std::vector<f_t>& solution,
                                     f_t zero_tol,
                                     simplex::logger_t& log);

template <typename i_t, typename f_t>
branch_variable_t<i_t> vector_length_diving(const simplex::lp_problem_t<i_t, f_t>& lp,
                                            const std::vector<i_t>& fractional,
                                            const std::vector<f_t>& solution,
                                            simplex::logger_t& log);

}  // namespace cuopt::mathematical_optimization::mip
