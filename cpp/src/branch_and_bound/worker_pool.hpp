/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/worker.hpp>

namespace cuopt::linear_programming::dual_simplex {

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

  void broadcast_root_bounds_change()
  {
    if (is_initialized) {
      for (i_t i = 0; i < workers_.size(); ++i) {
        if (workers_[i]->search_strategy == BEST_FIRST && workers_[i]->is_active) {
          workers_[i]->start_bounds_updated = true;
        }
      }
    }
  }

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