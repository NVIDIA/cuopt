/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace cuopt {

/**
 * One-way synchronization utility for producer threads.
 *
 * Producers (e.g., CPUFJ) register their work unit progress atomics and advance independently.
 * The consumer (e.g., B&B coordinator) can wait until all producers have reached a
 * target work unit threshold before proceeding.
 *
 * Key invariant: Producers must not fall behind the consumer's horizon. The consumer
 * waits at sync points until all producers have caught up. The producers are biased
 * to ensure they remain likely ahead of the consumer.
 */
class producer_sync_t {
 public:
  producer_sync_t() = default;

  void register_producer(std::atomic<double>* progress_ptr)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    producers_.push_back(progress_ptr);
    cv_.notify_all();
  }

  void deregister_producer(std::atomic<double>* progress_ptr)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(producers_.begin(), producers_.end(), progress_ptr);
    if (it != producers_.end()) { producers_.erase(it); }
    cv_.notify_all();
  }

  /**
   * Signal that all expected producers have been registered.
   * Must be called before the consumer can proceed with wait_for_producers().
   */
  void registration_complete()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    registration_complete_ = true;
    cv_.notify_all();
  }

  bool is_registration_complete() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return registration_complete_;
  }

  /**
   * Wait until:
   * 1. registration_complete() has been called, AND
   * 2. All registered producers have work units >= target_work_units
   *
   * Returns immediately if no producers are registered (after registration_complete).
   */
  void wait_for_producers(double target_work_units)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this, target_work_units] {
      if (!registration_complete_) { return false; }
      return all_producers_at_or_ahead(target_work_units);
    });
  }

  /**
   * Wake up any waiting consumer. Call this when a producer advances its work units.
   */
  void notify_progress() { cv_.notify_all(); }

  size_t num_producers() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return producers_.size();
  }

 private:
  bool all_producers_at_or_ahead(double target) const
  {
    for (const auto* progress_ptr : producers_) {
      if (progress_ptr->load(std::memory_order_acquire) < target) { return false; }
    }
    return true;
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<std::atomic<double>*> producers_;
  bool registration_complete_{false};
};

}  // namespace cuopt
