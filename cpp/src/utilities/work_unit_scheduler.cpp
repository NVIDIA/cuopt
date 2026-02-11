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

#include "work_unit_scheduler.hpp"

#include "work_limit_context.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

#include <omp.h>

#include <mip/logger.hpp>

namespace cuopt {

work_unit_scheduler_t::work_unit_scheduler_t(double sync_interval) : sync_interval_(sync_interval)
{
}

void work_unit_scheduler_t::register_context(work_limit_context_t& ctx)
{
  contexts_.push_back(ctx);
  ctx.scheduler = this;
}

void work_unit_scheduler_t::deregister_context(work_limit_context_t& ctx)
{
  ctx.scheduler = nullptr;
  contexts_.erase(std::remove_if(contexts_.begin(),
                                 contexts_.end(),
                                 [&ctx](const std::reference_wrapper<work_limit_context_t>& ref) {
                                   return &ref.get() == &ctx;
                                 }),
                  contexts_.end());
}

void work_unit_scheduler_t::set_sync_interval(double interval) { sync_interval_ = interval; }

void work_unit_scheduler_t::on_work_recorded(work_limit_context_t& ctx, double total_work)
{
  if (is_shutdown()) return;

  if (verbose) {
    CUOPT_LOG_DEBUG("[%s] Work recorded: %f, sync_target: %f (gen %zu)",
                    ctx.name.c_str(),
                    total_work,
                    current_sync_target(),
                    barrier_generation_);
  }

  // Loop to handle large work increments that cross multiple sync points
  while (total_work >= current_sync_target() && !is_shutdown()) {
    wait_at_sync_point(ctx, current_sync_target());
  }
}

void work_unit_scheduler_t::set_sync_callback(sync_callback_t callback)
{
  sync_callback_ = std::move(callback);
}

void work_unit_scheduler_t::wait_for_next_sync(work_limit_context_t& ctx)
{
  if (is_shutdown()) return;

  double next_sync              = current_sync_target();
  ctx.global_work_units_elapsed = next_sync;
  wait_at_sync_point(ctx, next_sync);
}

double work_unit_scheduler_t::current_sync_target() const
{
  if (sync_interval_ <= 0) return std::numeric_limits<double>::infinity();
  return (barrier_generation_ + 1) * sync_interval_;
}

void work_unit_scheduler_t::wait_at_sync_point(work_limit_context_t& ctx, double sync_target)
{
  auto wait_start = std::chrono::high_resolution_clock::now();

  if (verbose) {
    CUOPT_LOG_DEBUG("[%s] Waiting at sync point %.2f (gen %zu)",
                    ctx.name.c_str(),
                    sync_target,
                    barrier_generation_);
  }

  // All threads wait at this barrier
#pragma omp barrier

  // One thread executes the sync callback
#pragma omp single
  {
    current_sync_target_ = sync_target;
    barrier_generation_++;

    if (verbose) {
      CUOPT_LOG_DEBUG("All contexts arrived at sync point %.2f, new generation %zu",
                      sync_target,
                      barrier_generation_);
    }

    if (sync_callback_) { sync_callback_(sync_target); }
  }
  // Implicit barrier at end of single block ensures callback is complete
  // before any thread proceeds

  auto wait_end    = std::chrono::high_resolution_clock::now();
  double wait_secs = std::chrono::duration<double>(wait_end - wait_start).count();
  ctx.total_sync_time += wait_secs;

  if (verbose) {
    CUOPT_LOG_DEBUG("[%s] Sync complete at %.2f, waited %.2f ms",
                    ctx.name.c_str(),
                    sync_target,
                    wait_secs * 1000.0);
  }
}

}  // namespace cuopt
