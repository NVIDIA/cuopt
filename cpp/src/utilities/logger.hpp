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
 * Callers outside the libraries cannot reach a hidden logger, so each component exports a
 * configure entry point instead -- see log_target_t and init_component_logger_t below.
 */
namespace cuopt {

struct buffered_entry {
  rapids_logger::level_enum level;
  std::string msg;
};

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
    cuopt::default_logger().sinks().push_back(
      std::make_shared<rapids_logger::ostream_sink_mt>(std::cout));
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

// Ref-counted initializer for the logger of the image that constructs it. Library code uses
// this directly (routing configures routing's logger, mathopt configures mathopt's); callers
// outside the libraries should use init_component_logger_t to reach a library's instead.
class init_logger_t {
  std::shared_ptr<void> guard_;

 public:
  init_logger_t(std::string log_file, bool log_to_console, bool truncate = true);
};

inline std::mutex g_guard_mutex;

// Bumped for every configuration applied. A guard resets the logger only if its own
// configuration is still the current one.
inline uint64_t& active_config_generation()
{
  static uint64_t generation = 0;
  return generation;
}

// Guard whose destruction resets the logger, if its configuration is still current. The
// generation check matters: a guard's refcount reaching zero expires g_active_guard *before*
// this destructor runs, so another thread can install a new configuration in that window, and
// without the check this destructor would reset the logger out from under it.
struct logger_config_guard {
  explicit logger_config_guard(uint64_t generation) : generation_(generation) {}

  ~logger_config_guard()
  {
    std::lock_guard<std::mutex> lock(g_guard_mutex);
    if (active_config_generation() != generation_) { return; }
    cuopt::reset_default_logger();
  }

 private:
  uint64_t generation_;
};

// Weak reference to detect if any init_logger_t instance is still alive
inline std::weak_ptr<logger_config_guard> g_active_guard;

// Applies a configuration and returns a handle that keeps it alive. Single lifetime mechanism
// for this image's logger: init_logger_t and the exported per-component configure_logging both
// go through it, so a caller inside the library and one outside share one refcount, and the
// logger resets only when the last handle drops -- which is what stops the solver
// reconfiguring mid-run and re-truncating a log file the caller had already written to.
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

  auto guard     = std::make_shared<logger_config_guard>(++active_config_generation());
  g_active_guard = guard;
  return guard;
}

inline init_logger_t::init_logger_t(std::string log_file, bool log_to_console, bool truncate)
  : guard_(make_logger_config(log_file, log_to_console, truncate))
{
}

/**
 * @brief Which component library's logger to configure.
 */
enum class log_target_t {
  mathopt,  ///< LP / MILP / QP, in cuopt_mathopt
  routing   ///< VRP, in cuopt_routing
};

}  // namespace cuopt

// Exported per-component entry points. Each is defined in exactly one component library, and
// configures that library's own hidden logger -- the only logging symbols crossing a boundary.
namespace cuopt::mathematical_optimization {
CUOPT_EXPORT std::shared_ptr<void> configure_logging(const std::string& log_file,
                                                     bool log_to_console,
                                                     bool truncate);
}  // namespace cuopt::mathematical_optimization

#ifdef CUOPT_HAS_ROUTING
namespace cuopt::routing {
CUOPT_EXPORT std::shared_ptr<void> configure_logging(const std::string& log_file,
                                                     bool log_to_console,
                                                     bool truncate);
}  // namespace cuopt::routing
#endif

namespace cuopt {

// Configures a component library's logger from outside that library. The CLI, tests and
// Python bindings each hold their own logger and need to reach into the solver's; this
// dispatches to the component's exported entry point instead of init_logger_t. Defaults to
// mathopt because every external caller today is LP or MILP; routing is opted in explicitly.
class init_component_logger_t {
  // Same refcount init_logger_t uses, so a caller here and library code inside the component
  // cannot tear down each other's configuration.
  std::shared_ptr<void> handle_;

 public:
  explicit init_component_logger_t(const std::string& log_file,
                                   bool log_to_console,
                                   log_target_t target = log_target_t::mathopt,
                                   bool truncate       = true)
  {
    switch (target) {
      case log_target_t::routing:
#ifdef CUOPT_HAS_ROUTING
        handle_ = cuopt::routing::configure_logging(log_file, log_to_console, truncate);
#else
        // Silently doing nothing here would look like a working logger that drops every
        // message, and the cause -- a SKIP_ROUTING_BUILD mismatch -- would be invisible.
        throw std::runtime_error(
          "cuOpt was built with SKIP_ROUTING_BUILD, so routing's logger does not exist and "
          "log_target_t::routing cannot be configured.");
#endif
        break;
      case log_target_t::mathopt:
      default:
        handle_ =
          cuopt::mathematical_optimization::configure_logging(log_file, log_to_console, truncate);
        break;
    }
  }

  init_component_logger_t(const init_component_logger_t&)            = delete;
  init_component_logger_t& operator=(const init_component_logger_t&) = delete;
};

}  // namespace cuopt

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
