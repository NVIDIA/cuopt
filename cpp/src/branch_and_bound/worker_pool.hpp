/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/worker.hpp>
#include <utilities/circular_deque.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename WorkerType>
class worker_pool_t {
 public:
  using i_t = typename WorkerType::int_type;
  using f_t = typename WorkerType::float_type;

  void init(i_t num_workers,
            const lp_problem_t<i_t, f_t>& original_lp,
            const csr_matrix_t<i_t, f_t>& Arow,
            const std::vector<variable_type_t>& var_type,
            const simplex_solver_settings_t<i_t, f_t>& settings,
            const uint64_t rng_offset = 0)
  {
    assert(!is_initialized);
    assert(num_workers > 0);

    workers_.resize(num_workers);
    num_idle_workers_ = num_workers;
    idle_workers_.clear_resize(num_workers);
    for (i_t i = 0; i < num_workers; ++i) {
      workers_[i] =
        std::make_unique<WorkerType>(i, original_lp, Arow, var_type, settings, rng_offset);
      idle_workers_.push_back(i);
    }

    is_initialized = true;
  }

  WorkerType* pop_idle_worker()
  {
    std::lock_guard lock(mutex_);
    if (idle_workers_.empty()) {
      return nullptr;
    } else {
      i_t idx = idle_workers_.front();
      idle_workers_.pop_front();
      num_idle_workers_--;
      assert(idle_workers_.size() == static_cast<size_t>(num_idle_workers_.load()));
      assert(idx >= 0 && static_cast<size_t>(idx) < workers_.size());
      return workers_[idx].get();
    }
  }

  void return_worker_to_pool(WorkerType* worker)
  {
    std::lock_guard lock(mutex_);
    assert(worker != nullptr);
    assert(workers_[worker->worker_id].get() == worker);
    assert(!worker->is_active.load());
    assert(static_cast<size_t>(num_idle_workers_.load()) == idle_workers_.size());
    assert(idle_workers_.size() <= workers_.size());

    idle_workers_.push_back(worker->worker_id);
    num_idle_workers_++;
  }

  f_t get_lower_bound()
  {
    f_t lower_bound = std::numeric_limits<f_t>::infinity();

    if (is_initialized) {
      for (i_t i = 0; i < workers_.size(); ++i) {
        if (workers_[i]->is_active.load()) {
          lower_bound = std::min(workers_[i]->get_lower_bound(), lower_bound);
        }
      }
    }

    return lower_bound;
  }

  WorkerType* operator[](i_t id)
  {
    assert(id >= 0 && static_cast<size_t>(id) < workers_.size());
    assert(workers_[id] != nullptr);
    return workers_[id].get();
  }
  WorkerType* operator[](i_t id) const
  {
    assert(id >= 0 && static_cast<size_t>(id) < workers_.size());
    assert(workers_[id] != nullptr);
    return workers_[id].get();
  }

  i_t num_idle_workers() const { return num_idle_workers_; }
  i_t num_workers() const { return workers_.size(); }

 private:
  std::vector<std::unique_ptr<WorkerType>> workers_;
  bool is_initialized = false;

  omp_mutex_t mutex_;
  circular_deque_t<i_t> idle_workers_;
  omp_atomic_t<i_t> num_idle_workers_{0};
};

template <typename i_t, typename f_t>
using bfs_worker_pool_t = worker_pool_t<bfs_worker_t<i_t, f_t>>;

template <typename i_t, typename f_t>
using diving_worker_pool_t = worker_pool_t<diving_worker_t<i_t, f_t>>;

}  // namespace cuopt::linear_programming::dual_simplex
