/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/logger.hpp>

#include <mutex>

namespace cuopt {

namespace {

// A function-local static, not a namespace-scope global: std::mutex is not trivially
// destructible, and a namespace-scope instance would be torn down in an unspecified order
// relative to other static destructors at exit.
std::mutex& console_callback_mutex()
{
  static std::mutex mutex;
  return mutex;
}

log_console_callback_t g_console_callback = nullptr;
}  // namespace

void set_console_log_callback(log_console_callback_t callback)
{
  std::lock_guard<std::mutex> lock(console_callback_mutex());
  g_console_callback = callback;
}

log_console_callback_t console_log_callback()
{
  std::lock_guard<std::mutex> lock(console_callback_mutex());
  return g_console_callback;
}

}  // namespace cuopt
