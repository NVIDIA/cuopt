/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/export.hpp>

#include <cuopt/logger_macros.hpp>

#include <rapids_logger/logger.hpp>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace CUOPT_EXPORT cuopt {

/**
 * @brief Get the default logger.
 *
 * @return logger& The default logger
 */
rapids_logger::logger& default_logger();

/**
 * @brief Reset the default logger to the default settings.
 *  This is needed when we are running multiple tests and each test has different logger settings
 *  and we need to reset the logger to the default settings before each test.
 */
void reset_default_logger();

// C-compatible log callback type. Matches cuOptLogCallback in cuopt_c.h.
// Carries no severity by design: a level would become a de-facto public API.
using log_callback_with_data_t = void (*)(const char* message, void* user_data);

/**
 * @brief Registers a log callback for the calling thread, for its own lifetime.
 *
 * Registration is per-thread so concurrent solves cannot capture each other's
 * callback. Log lines emitted on other threads are not delivered.
 */
class scoped_log_callback_t {
 public:
  scoped_log_callback_t(log_callback_with_data_t cb, void* user_data);
  ~scoped_log_callback_t();

  scoped_log_callback_t(const scoped_log_callback_t&)            = delete;
  scoped_log_callback_t& operator=(const scoped_log_callback_t&) = delete;

 private:
  log_callback_with_data_t prev_callback_;
  void* prev_user_data_;
};

// Ref-counted logger initializer
class init_logger_t {
  // Using shared_ptr for ref-counting
  std::shared_ptr<void> guard_;

 public:
  init_logger_t(std::string log_file, bool log_to_console);
};

}  // namespace CUOPT_EXPORT cuopt

namespace cuopt::detail {

// Returns true for the first N calls sharing this counter.
template <auto N>
inline bool log_first_n_should_emit(std::atomic<uint64_t>& counter)
{
  static_assert(std::is_integral_v<decltype(N)>,
                "CUOPT_LOG_FIRST_N/CUOPT_LOG_ONCE requires an integral N");
  static_assert(N > 0, "CUOPT_LOG_FIRST_N/CUOPT_LOG_ONCE requires N > 0");
  constexpr uint64_t threshold = (uint64_t)N;

  if (counter.load(std::memory_order_relaxed) >= threshold) { return false; }
  return counter.fetch_add(1, std::memory_order_relaxed) < threshold;
}

// Returns true on calls 1, N+1, 2N+1, ...
template <auto N>
inline bool log_every_n_should_emit(std::atomic<uint64_t>& counter)
{
  static_assert(std::is_integral_v<decltype(N)>, "CUOPT_LOG_EVERY_N requires an integral N");
  static_assert(N > 0, "CUOPT_LOG_EVERY_N requires N > 0");
  return counter.fetch_add(1, std::memory_order_relaxed) % (uint64_t)N == 0;
}

}  // namespace cuopt::detail

// Rate-limited logging built on the generated CUOPT_LOG_<level> macros. `level` is one of
// TRACE/DEBUG/INFO/WARN/ERROR/CRITICAL; `n` must be a positive compile-time constant. Each
// call site owns its own counter, so throttling is independent per use.
#define CUOPT_LOG_FIRST_N(level, n, ...)                                   \
  do {                                                                     \
    static std::atomic<uint64_t> _cuopt_log_counter{0};                    \
    if (cuopt::detail::log_first_n_should_emit<(n)>(_cuopt_log_counter)) { \
      CUOPT_LOG_##level(__VA_ARGS__);                                      \
    }                                                                      \
  } while (0)

#define CUOPT_LOG_EVERY_N(level, n, ...)                                   \
  do {                                                                     \
    static std::atomic<uint64_t> _cuopt_log_counter{0};                    \
    if (cuopt::detail::log_every_n_should_emit<(n)>(_cuopt_log_counter)) { \
      CUOPT_LOG_##level(__VA_ARGS__);                                      \
    }                                                                      \
  } while (0)

#define CUOPT_LOG_ONCE(level, ...) CUOPT_LOG_FIRST_N(level, 1, __VA_ARGS__)
