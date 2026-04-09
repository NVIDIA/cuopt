/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <algorithm>
#include <cmath>
#include <string>

#include <mip_heuristics/logger.hpp>

#include "copy_helpers.hpp"
#include "timer.hpp"
#include "work_limit_context.hpp"

namespace cuopt {

/**
 * Unified termination checker that combines timer_t and termination_checker_t.
 *
 * In non-deterministic mode: checks wall-clock time.
 * In deterministic mode: checks work units via work_limit_context_t.
 * In BOTH modes: checks parent chain (inheriting root wall-clock limit) and user callbacks.
 */
class termination_checker_t {
 public:
  struct root_tag_t {};

  // Root constructor (top-level solver, wall-clock only)
  explicit termination_checker_t(double time_limit, root_tag_t)
    : deterministic(false),
      work_limit(time_limit),
      timer(time_limit),
      work_context(nullptr),
      work_units_at_start(0),
      parent_(nullptr)
  {
  }

  // Non-deterministic constructor with parent
  termination_checker_t(double time_limit_, const termination_checker_t& parent)
    : deterministic(false),
      work_limit(time_limit_),
      timer(time_limit_),
      work_context(nullptr),
      work_units_at_start(0),
      parent_(&parent)
  {
  }

  // Deterministic constructor with parent (inherits parent's termination)
  termination_checker_t(work_limit_context_t& context,
                        double work_limit_,
                        const termination_checker_t& parent)
    : deterministic(context.deterministic),
      work_limit(work_limit_),
      timer(work_limit_),
      work_context(&context),
      work_units_at_start(context.deterministic ? context.current_work() : 0),
      parent_(&parent)
  {
  }

  void set_parent(const termination_checker_t* parent) { parent_ = parent; }
  const termination_checker_t* get_parent() const { return parent_; }

  void set_termination_callback(bool (*cb)(void*), void* data)
  {
    termination_callback_      = cb;
    termination_callback_data_ = data;
  }

  bool check(const char* caller = __builtin_FUNCTION(),
             const char* file   = __builtin_FILE(),
             int line           = __builtin_LINE()) const noexcept
  {
    if (termination_callback_ != nullptr && termination_callback_(termination_callback_data_)) {
      return true;
    }

    if (parent_ != nullptr && parent_->check()) { return true; }

    if (deterministic) {
      if (!work_context) { return false; }
      double elapsed_since_start = work_context->current_work() - work_units_at_start;
      bool finished_now          = elapsed_since_start >= work_limit;
      if (finished_now && !finished) {
        finished                   = true;
        double actual_elapsed_time = timer.elapsed_time();

        if (work_limit > 0 && std::abs(actual_elapsed_time - work_limit) / work_limit > 0.10) {
          CUOPT_LOG_TRACE(
            "%s:%d: %s(): Work limit timer finished with a large discrepancy: %fs for %fwu "
            "(global: %g, start: %g)",
            file,
            line,
            caller,
            actual_elapsed_time,
            work_limit,
            work_context->current_work(),
            work_units_at_start);
        }
      }
      return finished;
    } else {
      return timer.check_time_limit();
    }
  }

  // Aliases for compatibility with termination_checker_t and timer_t interfaces
  bool check_time_limit(const char* caller = __builtin_FUNCTION(),
                        const char* file   = __builtin_FILE(),
                        int line           = __builtin_LINE()) const noexcept
  {
    return check(caller, file, line);
  }

  bool check_limit(const char* caller = __builtin_FUNCTION(),
                   const char* file   = __builtin_FILE(),
                   int line           = __builtin_LINE()) const noexcept
  {
    return check(caller, file, line);
  }

  void record_work(double work_units,
                   const char* caller = __builtin_FUNCTION(),
                   const char* file   = __builtin_FILE(),
                   int line           = __builtin_LINE())
  {
    if (deterministic && work_context) {
      // debugging info
      double parent_elapsed_time = parent_ != nullptr ? parent_->timer.elapsed_time() : 0.0;
      double parent_time_limit   = parent_ != nullptr ? parent_->timer.get_time_limit() : 0.0;

      CUOPT_LOG_TRACE("%s:%d: %s(): Recorded %f work units in %fs, total %f (parent time: %g/%g)",
                      file,
                      line,
                      caller,
                      work_units,
                      timer.elapsed_time(),
                      work_context->current_work(),
                      parent_elapsed_time,
                      parent_time_limit);
      work_context->record_work_sync_on_horizon(work_units);
    }
  }

  double remaining_units() const noexcept
  {
    double local_remaining;
    if (deterministic) {
      if (!work_context) {
        local_remaining = work_limit;
      } else {
        double elapsed_since_start = work_context->current_work() - work_units_at_start;
        local_remaining            = std::max(0.0, work_limit - elapsed_since_start);
      }
    } else {
      local_remaining = timer.remaining_time();
    }
    // don't let the root's global time limit contaminate work limits further down
    if (parent_ != nullptr && !(deterministic && !parent_->deterministic)) {
      local_remaining = std::min(local_remaining, parent_->remaining_units());
    }
    if (!std::isfinite(local_remaining)) {
      CUOPT_LOG_WARN(
        "remaining_units non-finite: %g det=%d work_limit=%g start=%g "
        "ctx_work=%g has_parent=%d",
        local_remaining,
        (int)deterministic,
        work_limit,
        work_units_at_start,
        work_context ? work_context->current_work() : -1.0,
        parent_ != nullptr);
    }
    return local_remaining;
  }

  double remaining_time() const noexcept { return remaining_units(); }

  double elapsed_time() const noexcept
  {
    if (deterministic) {
      if (!work_context) { return 0.0; }
      return work_context->current_work() - work_units_at_start;
    } else {
      return timer.elapsed_time();
    }
  }

  bool check_half_time() const noexcept
  {
    if (deterministic) {
      if (!work_context) { return false; }
      double elapsed_since_start = work_context->current_work() - work_units_at_start;
      return elapsed_since_start >= work_limit / 2;
    } else {
      return timer.check_half_time();
    }
  }

  double clamp_remaining_time(double desired_time) const noexcept
  {
    return std::min<double>(desired_time, remaining_time());
  }

  double get_time_limit() const noexcept
  {
    if (deterministic) {
      return work_limit;
    } else {
      return timer.get_time_limit();
    }
  }

  double get_tic_start() const noexcept { return timer.get_tic_start(); }

  timer_t timer;
  double work_limit{};
  mutable copyable_atomic_t<bool> finished{false};
  bool deterministic{false};
  work_limit_context_t* work_context{nullptr};
  double work_units_at_start{0};

 private:
  const termination_checker_t* parent_{nullptr};
  bool (*termination_callback_)(void*) = nullptr;
  void* termination_callback_data_     = nullptr;
};

}  // namespace cuopt
