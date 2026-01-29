/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <chrono>
#include <string>

namespace cuopt {

// TODO extend this for the whole solver.
// we are currently using this for diversity and adapters
class timer_t {
  using steady_clock = std::chrono::steady_clock;

 public:
  timer_t()               = delete;
  timer_t(const timer_t&) = default;
  timer_t(double time_limit_)
  {
    time_limit = time_limit_;
    begin      = steady_clock::now();
  }

  void print_debug(std::string msg) const
  {
    printf("%s time_limit: %f remaining_time: %f elapsed_time: %f \n",
           msg.c_str(),
           time_limit,
           remaining_time(),
           elapsed_time());
  }

  bool check_time_limit() const noexcept { return elapsed_time() >= time_limit; }

  bool check_half_time() const noexcept { return elapsed_time() >= time_limit / 2; }

  double elapsed_time() const noexcept
  {
    return std::chrono::duration<double>(steady_clock::now() - begin).count();
  }

  double remaining_time() const noexcept
  {
    return std::max<double>(0.0, time_limit - elapsed_time());
  }

  double clamp_remaining_time(double desired_time) const noexcept
  {
    return std::min<double>(desired_time, remaining_time());
  }

  double get_time_limit() const noexcept { return time_limit; }

  double get_tic_start() const noexcept
  {
    /**
     * Converts a std::chrono::steady_clock::time_point to a struct timeval.
     * This is an approximate conversion because steady_clock is relative to an
     * unspecified epoch (e.g., system boot time), not the system clock epoch (UTC).
     */
    // Get the current time from both clocks at approximately the same instant
    std::chrono::system_clock::time_point sys_now    = std::chrono::system_clock::now();
    std::chrono::steady_clock::time_point steady_now = std::chrono::steady_clock::now();

    // Calculate the difference between the given steady_clock time point and the current steady
    // time
    auto diff_from_now = begin - steady_now;

    // Apply that same difference to the current system clock time point
    std::chrono::system_clock::time_point sys_t = sys_now + diff_from_now;

    // Convert the resulting system_clock time point to microseconds since the system epoch
    auto us_since_epoch =
      std::chrono::duration_cast<std::chrono::microseconds>(sys_t.time_since_epoch());

    // Populate the timeval struct
    double tv_sec  = us_since_epoch.count() / 1000000;
    double tv_usec = us_since_epoch.count() % 1000000;

    return tv_sec + 1e-6 * tv_usec;
  }

 private:
  double time_limit;
  steady_clock::time_point begin;
};

}  // namespace cuopt
