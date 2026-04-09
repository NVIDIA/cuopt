/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/worker.hpp>
#include <utilities/lock_free_stack.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename WorkerType>
class worker_pool_t {
 public:
  using i_t = WorkerType::int_type;
  using f_t = WorkerType::float_type;

  void init(i_t num_workers,
            const lp_problem_t<i_t, f_t>& original_lp,
            const csr_matrix_t<i_t, f_t>& Arow,
            const std::vector<variable_type_t>& var_type,
            const simplex_solver_settings_t<i_t, f_t>& settings,
            const uint64_t rng_offset = 0)
  {
    workers_.resize(num_workers);
    idle_stack_.init(num_workers);
    for (i_t i = 0; i < num_workers; ++i) {
      workers_[i] =
        std::make_unique<WorkerType>(i, original_lp, Arow, var_type, settings, rng_offset);
      idle_stack_.push(i);
    }

    is_initialized = true;
  }

  WorkerType* pop_idle_worker()
  {
    std::optional<i_t> idx = idle_stack_.pop();
    return idx ? workers_[*idx].get() : nullptr;
  }

  void return_worker_to_pool(WorkerType* worker)
  {
    worker->is_active = false;
    idle_stack_.push(worker->worker_id);
  }

  f_t get_lower_bound()
  {
    f_t lower_bound = std::numeric_limits<f_t>::infinity();

    if (is_initialized) {
      for (i_t i = 0; i < workers_.size(); ++i) {
        lower_bound = std::min(workers_[i]->get_lower_bound(), lower_bound);
      }
    }

    return lower_bound;
  }

  WorkerType* get_worker(i_t id) { return workers_[id].get(); }

  std::vector<WorkerType*> get_active_workers()
  {
    std::vector<WorkerType*> active_workers;
    for (i_t i = 0; i < workers_.size(); ++i) {
      if (workers_[i]->is_active) { active_workers.push_back(workers_[i].get()); }
    }

    return active_workers;
  }

  i_t num_idle_workers() const { return idle_stack_.size(); }
  i_t num_workers() const { return workers_.size(); }

 private:
  // Worker pool
  std::vector<std::unique_ptr<WorkerType>> workers_;
  bool is_initialized = false;

  lock_free_index_stack_t<i_t> idle_stack_;
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

template <typename i_t, typename f_t>
using bfs_worker_pool_t = worker_pool_t<bfs_worker_t<i_t, f_t>>;

template <typename i_t, typename f_t>
using diving_worker_pool_t = worker_pool_t<diving_worker_t<i_t, f_t>>;

}  // namespace cuopt::linear_programming::dual_simplex
