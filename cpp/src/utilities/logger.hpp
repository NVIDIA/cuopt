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
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

/*
 * Defined inline with hidden visibility so each library that links this header owns its own
 * logger instance. This is not optional: an inline function's static local is emitted as an
 * STB_GNU_UNIQUE symbol, which glibc merges process-wide regardless of RTLD_LOCAL, so default
 * visibility would collapse every library back into one shared logger. Do not mark this
 * namespace CUOPT_EXPORT.
 *
 * Each solver already builds an init_logger_t from its own settings on entry, so passing a
 * log file through settings configures that library's logger. An executable linking cuopt
 * has a separate logger for its own messages, and when both write the same file the solver
 * would truncate it mid-solve and discard what the executable had already written. The one
 * exported entry point below lets the executable establish the configuration first, so the
 * solver's own initializer reuses it instead of replacing it.
 */
namespace cuopt {

struct buffered_entry {
  rapids_logger::level_enum level;
  std::string msg;
};

using log_console_callback_t = void (*)(int level, const char* message);

inline std::mutex g_console_callback_mutex;
inline log_console_callback_t g_console_callback = nullptr;

/**
 * @brief Overrides the sink used for console logging (settings.log_to_console == true).
 *
 * Passing nullptr (the default) restores writing to std::cout. Intended for language bindings
 * whose host runtime cannot safely receive a raw write to the native stdout stream.
 *
 * Per-image state, like the logger itself -- reach a specific component library's copy through
 * its exported `set_console_log_callback`, the same way `configure_logging` reaches its logger.
 *
 * @param callback The callback to invoke for each logged line, or nullptr to restore std::cout.
 */
inline void set_console_log_callback(log_console_callback_t callback)
{
  std::lock_guard<std::mutex> lock(g_console_callback_mutex);
  g_console_callback = callback;
}

inline log_console_callback_t console_log_callback()
{
  std::lock_guard<std::mutex> lock(g_console_callback_mutex);
  return g_console_callback;
}

// Buffer to store log messages
class log_buffer {
 public:
  log_buffer()  = default;
  ~log_buffer() = default;

  void log(rapids_logger::level_enum lvl, const char* msg)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!msg) return;
    std::string str(msg);

    if (!str.empty() && str.back() == '\n') { str.pop_back(); }
    messages.push_back({lvl, std::move(str)});
  }

  size_t size() const
  {
    std::lock_guard<std::mutex> lock(mutex);
    return messages.size();
  }

  std::vector<buffered_entry> drain_all()
  {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<buffered_entry> out;
    out.swap(messages);
    return out;
  }

 private:
  std::vector<buffered_entry> messages;
  mutable std::mutex mutex;
};

inline log_buffer& global_log_buffer()
{
  static log_buffer buffer;
  return buffer;
}

inline void buffer_log_callback(int lvl, const char* msg)
{
  global_log_buffer().log(static_cast<rapids_logger::level_enum>(lvl), msg);
}

// Buffers messages in memory until something configures the logger; anything logged before
// that, and never followed by a configure, is dropped.
inline rapids_logger::sink_ptr default_sink()
{
  return std::make_shared<rapids_logger::callback_sink_mt>(buffer_log_callback);
}

inline std::string default_pattern() { return "[%Y-%m-%d %H:%M:%S:%f] [%n] [%-6l] %v"; }

inline rapids_logger::level_enum default_level()
{
#if CUOPT_LOG_ACTIVE_LEVEL == RAPIDS_LOGGER_LOG_LEVEL_TRACE
  return rapids_logger::level_enum::trace;
#elif CUOPT_LOG_ACTIVE_LEVEL == RAPIDS_LOGGER_LOG_LEVEL_DEBUG
  return rapids_logger::level_enum::debug;
#elif CUOPT_LOG_ACTIVE_LEVEL == RAPIDS_LOGGER_LOG_LEVEL_INFO
  return rapids_logger::level_enum::info;
#elif CUOPT_LOG_ACTIVE_LEVEL == RAPIDS_LOGGER_LOG_LEVEL_WARN
  return rapids_logger::level_enum::warn;
#elif CUOPT_LOG_ACTIVE_LEVEL == RAPIDS_LOGGER_LOG_LEVEL_ERROR
  return rapids_logger::level_enum::error;
#elif CUOPT_LOG_ACTIVE_LEVEL == RAPIDS_LOGGER_LOG_LEVEL_CRITICAL
  return rapids_logger::level_enum::critical;
#else
  return rapids_logger::level_enum::info;
#endif
}

inline rapids_logger::logger& default_logger()
{
  static rapids_logger::logger logger_ = [] {
    rapids_logger::logger logger_{"CUOPT", {default_sink()}};
#if CUOPT_LOG_ACTIVE_LEVEL >= RAPIDS_LOGGER_LOG_LEVEL_INFO
    logger_.set_pattern("%v");
#else
    logger_.set_pattern(default_pattern());
#endif
    logger_.set_level(default_level());
    logger_.flush_on(rapids_logger::level_enum::debug);

    return logger_;
  }();

  return logger_;
}

inline void reset_default_logger()
{
  default_logger().sinks().clear();
  default_logger().sinks().push_back(default_sink());
#if CUOPT_LOG_ACTIVE_LEVEL >= RAPIDS_LOGGER_LOG_LEVEL_INFO
  default_logger().set_pattern("%v");
#else
  default_logger().set_pattern(default_pattern());
#endif
  default_logger().set_level(default_level());
  default_logger().flush_on(rapids_logger::level_enum::debug);
}

// Points this image's logger at the given sinks and flushes anything buffered so far.
// `truncate` clears log_file up front instead of letting the sink truncate it: several
// loggers in one process can share a path, and a truncating sink writes from offset 0,
// silently overwriting whatever another one has already appended. Pass false when another
// image is already logging to the same path and has truncated it.
inline void apply_logger_config(const std::string& log_file, bool log_to_console, bool truncate)
{
  cuopt::default_logger().sinks().clear();

  if (log_to_console) {
    if (auto callback = console_log_callback(); callback != nullptr) {
      cuopt::default_logger().sinks().push_back(
        std::make_shared<rapids_logger::callback_sink_mt>(callback));
    } else {
      cuopt::default_logger().sinks().push_back(
        std::make_shared<rapids_logger::ostream_sink_mt>(std::cout));
    }
  }
  if (!log_file.empty()) {
    if (truncate) { std::ofstream(log_file, std::ios::trunc); }
    cuopt::default_logger().sinks().push_back(
      std::make_shared<rapids_logger::basic_file_sink_mt>(log_file, /*truncate=*/false));
    cuopt::default_logger().flush_on(rapids_logger::level_enum::debug);
  }

#if CUOPT_LOG_ACTIVE_LEVEL >= RAPIDS_LOGGER_LOG_LEVEL_INFO
  cuopt::default_logger().set_pattern("%v");
#else
  cuopt::default_logger().set_pattern(cuopt::default_pattern());
#endif

  auto buffered_messages = global_log_buffer().drain_all();
  for (const auto& entry : buffered_messages) {
    cuopt::default_logger().log(entry.level, entry.msg.c_str());
  }
}

// Ref-counted initializer for the logger of the image that constructs it: constructed inside
// cuopt_routing it configures routing's logger, inside cuopt_mathopt it configures mathopt's,
// and in an executable it configures that executable's own.
class init_logger_t {
  std::shared_ptr<void> guard_;

 public:
  init_logger_t(std::string log_file, bool log_to_console, bool truncate = true);
};

inline std::mutex g_guard_mutex;

// Guard object whose destructor resets the logger
struct logger_config_guard {
  ~logger_config_guard() { cuopt::reset_default_logger(); }
};

// Weak reference to detect if any init_logger_t instance is still alive
inline std::weak_ptr<logger_config_guard> g_active_guard;

// Applies a configuration and returns a handle that keeps it alive. The logger resets only
// when the last handle drops, so nested initializers share one configuration -- which is what
// stops an inner solver reconfiguring mid-run and re-truncating a log file the outer caller
// had already written to.
inline std::shared_ptr<void> make_logger_config(const std::string& log_file,
                                                bool log_to_console,
                                                bool truncate)
{
  std::lock_guard<std::mutex> lock(g_guard_mutex);

  // Reuse the configuration already in place; reconfiguring here would re-truncate the file.
  if (auto existing = g_active_guard.lock()) { return existing; }

  try {
    apply_logger_config(log_file, log_to_console, truncate);
  } catch (...) {
    // Sinks are cleared before new ones install, so a throw here would otherwise leave the
    // logger with none at all.
    reset_default_logger();
    throw;
  }

  auto guard     = std::make_shared<logger_config_guard>();
  g_active_guard = guard;
  return guard;
}

inline init_logger_t::init_logger_t(std::string log_file, bool log_to_console, bool truncate)
  : guard_(make_logger_config(log_file, log_to_console, truncate))
{
}

}  // namespace cuopt

// Configures cuopt_mathopt's logger. The only logging symbols that cross a library boundary.
// configure_logging exists for one caller: an executable that writes the same log file as the
// solver and must configure it before the solver's own initializer would truncate it.
// set_console_log_callback exists for another: a language binding, such as Java, whose host
// runtime cannot safely receive a raw write to the native stdout stream.
namespace cuopt::mathematical_optimization {
CUOPT_EXPORT std::shared_ptr<void> configure_logging(const std::string& log_file,
                                                     bool log_to_console,
                                                     bool truncate);
CUOPT_EXPORT void set_console_log_callback(log_console_callback_t callback);
}  // namespace cuopt::mathematical_optimization

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
