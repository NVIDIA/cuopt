/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt/mathematical_optimization/constants.h>
#include "timer.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace cuopt {

/**
 * @brief Unified early-exit reasons polled by solvers (alongside optimality checks).
 *
 * Priority when multiple apply: Cancelled > ConcurrentHalt > TimeLimit > IterationLimit.
 */
enum class solve_limit_reason_t : int8_t {
  None           = 0,
  Cancelled      = 1,
  ConcurrentHalt = 2,
  TimeLimit      = 3,
  IterationLimit = 4,
};

namespace detail {

inline bool atomic_flag_set(const std::atomic<bool>* flag) noexcept
{
  return flag != nullptr && flag->load(std::memory_order_acquire);
}

inline bool concurrent_halt_set(const std::atomic<int>* halt) noexcept
{
  return halt != nullptr && halt->load(std::memory_order_acquire) == 1;
}

}  // namespace detail

/**
 * @brief Check cancel / concurrent halt / time / iteration budgets.
 *
 * Timer overload used by PDLP and MIP heuristics (`timer_t`).
 */
inline solve_limit_reason_t check_solve_limits(
  const timer_t& timer,
  const std::atomic<bool>* cancel_requested   = nullptr,
  const std::atomic<int>* concurrent_halt     = nullptr,
  std::optional<std::int64_t> iterations      = std::nullopt,
  std::optional<std::int64_t> iteration_limit = std::nullopt) noexcept
{
  const std::atomic<bool>* cancel =
    cancel_requested != nullptr ? cancel_requested : timer.get_cancel_requested();
  if (detail::atomic_flag_set(cancel)) { return solve_limit_reason_t::Cancelled; }
  if (detail::concurrent_halt_set(concurrent_halt)) { return solve_limit_reason_t::ConcurrentHalt; }
  if (timer.time_exhausted()) { return solve_limit_reason_t::TimeLimit; }
  if (iterations.has_value() && iteration_limit.has_value() &&
      iterations.value() >= iteration_limit.value()) {
    return solve_limit_reason_t::IterationLimit;
  }
  return solve_limit_reason_t::None;
}

/**
 * @brief Same checks using tic/toc wall-clock style (barrier, dual simplex, B&B).
 *
 * @param elapsed_seconds  Elapsed time, e.g. `toc(start_time)`.
 * @param time_limit       Wall-clock budget in seconds.
 */
inline solve_limit_reason_t check_solve_limits(
  double elapsed_seconds,
  double time_limit,
  const std::atomic<bool>* cancel_requested   = nullptr,
  const std::atomic<int>* concurrent_halt     = nullptr,
  std::optional<std::int64_t> iterations      = std::nullopt,
  std::optional<std::int64_t> iteration_limit = std::nullopt) noexcept
{
  if (detail::atomic_flag_set(cancel_requested)) { return solve_limit_reason_t::Cancelled; }
  if (detail::concurrent_halt_set(concurrent_halt)) { return solve_limit_reason_t::ConcurrentHalt; }
  if (elapsed_seconds > time_limit) { return solve_limit_reason_t::TimeLimit; }
  if (iterations.has_value() && iteration_limit.has_value() &&
      iterations.value() >= iteration_limit.value()) {
    return solve_limit_reason_t::IterationLimit;
  }
  return solve_limit_reason_t::None;
}

inline bool cancel_or_halt_requested(const std::atomic<bool>* cancel = nullptr,
                                     const std::atomic<int>* halt    = nullptr) noexcept
{
  return detail::atomic_flag_set(cancel) || detail::concurrent_halt_set(halt);
}

/**
 * @brief Convenience bool for loops that previously used only `timer.check_time_limit()`.
 *
 * If `cancel` is null, uses the cancel pointer embedded on `timer` (if any).
 */
inline bool solve_limit_reached(const timer_t& timer,
                                const std::atomic<bool>* cancel = nullptr,
                                const std::atomic<int>* halt    = nullptr) noexcept
{
  return check_solve_limits(timer, cancel, halt) != solve_limit_reason_t::None;
}

inline bool solve_limit_reached(double elapsed_seconds,
                                double time_limit,
                                const std::atomic<bool>* cancel = nullptr,
                                const std::atomic<int>* halt    = nullptr) noexcept
{
  return check_solve_limits(elapsed_seconds, time_limit, cancel, halt) !=
         solve_limit_reason_t::None;
}

/** Map to public LP/MIP termination constant values where defined. */
inline int solve_limit_to_termination_status(solve_limit_reason_t reason) noexcept
{
  switch (reason) {
    case solve_limit_reason_t::Cancelled: return CUOPT_TERMINATION_STATUS_CANCELLED;
    case solve_limit_reason_t::ConcurrentHalt: return CUOPT_TERMINATION_STATUS_CONCURRENT_LIMIT;
    case solve_limit_reason_t::TimeLimit: return CUOPT_TERMINATION_STATUS_TIME_LIMIT;
    case solve_limit_reason_t::IterationLimit: return CUOPT_TERMINATION_STATUS_ITERATION_LIMIT;
    case solve_limit_reason_t::None: return CUOPT_TERMINATION_STATUS_NO_TERMINATION;
  }
  return CUOPT_TERMINATION_STATUS_NO_TERMINATION;
}

inline bool cancel_flag_set(const std::atomic<bool>* cancel) noexcept
{
  return detail::atomic_flag_set(cancel);
}

/**
 * @brief Remap limit-like statuses to Cancelled when the cancel flag is set.
 *
 * Call at solution finalization so inner loops that collapse cancel into
 * TIME_LIMIT / CONCURRENT_LIMIT still surface Cancelled to callers.
 */
template <typename Status>
inline Status remap_limit_status_if_cancelled(const std::atomic<bool>* cancel,
                                              Status status,
                                              Status cancelled_status,
                                              Status time_limit_status,
                                              Status concurrent_limit_status) noexcept
{
  if (!detail::atomic_flag_set(cancel)) { return status; }
  if (status == time_limit_status || status == concurrent_limit_status ||
      status == cancelled_status) {
    return cancelled_status;
  }
  return status;
}

template <typename Status>
inline Status remap_limit_status_if_cancelled(const std::atomic<bool>* cancel,
                                              Status status,
                                              Status cancelled_status,
                                              Status time_limit_status,
                                              Status concurrent_limit_status,
                                              Status iteration_limit_status) noexcept
{
  if (!detail::atomic_flag_set(cancel)) { return status; }
  if (status == time_limit_status || status == concurrent_limit_status ||
      status == cancelled_status || status == iteration_limit_status) {
    return cancelled_status;
  }
  return status;
}

/**
 * @brief Optional helper to clear a cancel flag after a top-level solve returns.
 *
 * gRPC clears via job-slot reset instead so the worker can still observe the
 * flag after solve_* returns. Standalone callers may use this RAII guard.
 */
struct cancel_flag_clear_on_exit_t {
  std::atomic<bool>* flag{nullptr};
  explicit cancel_flag_clear_on_exit_t(std::atomic<bool>* f) noexcept : flag(f) {}
  cancel_flag_clear_on_exit_t(const cancel_flag_clear_on_exit_t&)            = delete;
  cancel_flag_clear_on_exit_t& operator=(const cancel_flag_clear_on_exit_t&) = delete;
  ~cancel_flag_clear_on_exit_t()
  {
    if (flag != nullptr) { flag->store(false, std::memory_order_release); }
  }
};

}  // namespace cuopt
