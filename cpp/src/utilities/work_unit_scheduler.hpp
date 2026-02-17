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

#include <atomic>
#include <functional>
#include <vector>

namespace cuopt {

struct work_limit_context_t;

class work_unit_scheduler_t {
 public:
  explicit work_unit_scheduler_t(double sync_interval = 5.0);

  void set_sync_interval(double interval);
  double get_sync_interval() const { return sync_interval_; }

  void register_context(work_limit_context_t& ctx);
  void deregister_context(work_limit_context_t& ctx);
  void on_work_recorded(work_limit_context_t& ctx, double total_work);

  // Sync callback - executed by one thread when all contexts reach sync point
  using sync_callback_t = std::function<void(double sync_target)>;
  void set_sync_callback(sync_callback_t callback);

  // Wait for next sync point (for idle workers with no work)
  void wait_for_next_sync(work_limit_context_t& ctx);

  double current_sync_target() const;

  void signal_shutdown() { shutdown_.store(true, std::memory_order_release); }
  bool is_shutdown() const { return shutdown_.load(std::memory_order_acquire); }

 public:
  bool verbose{false};

 private:
  void wait_at_sync_point(work_limit_context_t& ctx, double sync_target);

  double sync_interval_;
  std::vector<std::reference_wrapper<work_limit_context_t>> contexts_;

  size_t barrier_generation_{0};
  double current_sync_target_{0};

  // Sync callback - executed when all contexts reach sync point
  sync_callback_t sync_callback_;

  // Shutdown flag - prevents threads from entering barriers after termination is signaled
  std::atomic<bool> shutdown_{false};
};

// RAII helper for registering multiple contexts with automatic cleanup
class scoped_context_registrations_t {
 public:
  explicit scoped_context_registrations_t(work_unit_scheduler_t& scheduler) : scheduler_(scheduler)
  {
  }

  ~scoped_context_registrations_t()
  {
    for (auto* ctx : contexts_) {
      scheduler_.deregister_context(*ctx);
    }
  }

  void add(work_limit_context_t& ctx)
  {
    scheduler_.register_context(ctx);
    contexts_.push_back(&ctx);
  }

  scoped_context_registrations_t(const scoped_context_registrations_t&)            = delete;
  scoped_context_registrations_t& operator=(const scoped_context_registrations_t&) = delete;
  scoped_context_registrations_t(scoped_context_registrations_t&&)                 = delete;
  scoped_context_registrations_t& operator=(scoped_context_registrations_t&&)      = delete;

 private:
  work_unit_scheduler_t& scheduler_;
  std::vector<work_limit_context_t*> contexts_;
};

}  // namespace cuopt
