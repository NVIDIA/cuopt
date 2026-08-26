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
 * The logger and its buffer are defined inline and with hidden visibility, so each library
 * that links this header owns its own. cuOpt ships as separate solver libraries and
 * rapids_logger provides the logger type rather than a shared instance, so there is no
 * single place to host one without a library existing purely to hold it. Each solver
 * configures its own logging through its own settings.
 *
 * Hidden visibility is what does the separating, and it is not optional. The static local
 * of an inline function is emitted as an STB_GNU_UNIQUE symbol, which glibc merges across
 * the whole process regardless of RTLD_LOCAL, so a header-only logger with default
 * visibility would still be one shared instance. Do not mark this namespace CUOPT_EXPORT.
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

  std::vector<buffered_entry> messages;
  mutable std::mutex mutex;
};

inline log_buffer& global_log_buffer()
{
  static log_buffer buffer;
  return buffer;
}

// Callback function for the buffer sink
inline void buffer_log_callback(int lvl, const char* msg)
{
  // store level with message; actual filtering happens at logger time
  global_log_buffer().log(static_cast<rapids_logger::level_enum>(lvl), msg);
}

/**
 * @brief Returns the default sink, used until something configures the logger.
 *
 * Messages go into an in-memory buffer rather than to a stream, and are replayed once a
 * configuration arrives. Anything logged before that, and never followed by a configure,
 * is dropped.
 *
 * @return sink_ptr The sink to use
 */
inline rapids_logger::sink_ptr default_sink()
{
  return std::make_shared<rapids_logger::callback_sink_mt>(buffer_log_callback);
}

/**
 * @brief Returns the default log pattern for the global logger.
 *
 * @return std::string The default log pattern.
 */
inline std::string default_pattern() { return "[%Y-%m-%d %H:%M:%S:%f] [%n] [%-6l] %v"; }

/**
 * @brief Returns the default log level for the global logger.
 *
 * @return rapids_logger::level_enum The default log level.
 */
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

/**
 * @brief Point this image's logger at the given sinks and flush anything buffered so far.
 *
 * @param log_file       File to log to, or empty for none.
 * @param log_to_console Whether to also log to stdout.
 * @param truncate       Whether opening @p log_file clears it. Pass false when another
 *                       image is already logging to the same path and has truncated it.
 */
inline void apply_logger_config(const std::string& log_file, bool log_to_console, bool truncate)
{
  cuopt::default_logger().sinks().clear();

  // re-initialize sinks
  if (log_to_console) {
    cuopt::default_logger().sinks().push_back(
      std::make_shared<rapids_logger::ostream_sink_mt>(std::cout));
  }
  if (!log_file.empty()) {
    // Clear the file up front rather than letting the sink truncate. Several loggers in one
    // process can share a path -- the CLI has its own and the solver library has another --
    // and a truncating sink writes from offset 0, silently overwriting whatever the other
    // one has already appended. Opening every sink in append mode keeps them interleaving.
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

  // Extract messages from the global buffer and log to the default logger
  auto buffered_messages = global_log_buffer().drain_all();
  for (const auto& entry : buffered_messages) {
    cuopt::default_logger().log(entry.level, entry.msg.c_str());
  }
}

/**
 * @brief Ref-counted initializer for the logger of the image that constructs it.
 *
 * Library code uses this directly: constructed inside cuopt_routing it configures routing's
 * logger, inside cuopt_mathopt it configures mathopt's. Callers outside the libraries get
 * their own logger this way and should use init_component_logger_t to reach a library's.
 */
class init_logger_t {
  // Using shared_ptr for ref-counting
  std::shared_ptr<void> guard_;

 public:
  init_logger_t(std::string log_file, bool log_to_console, bool truncate = true);
};

// Guard object whose destructor resets the logger
struct logger_config_guard {
  ~logger_config_guard() { cuopt::reset_default_logger(); }
};

// Weak reference to detect if any init_logger_t instance is still alive
inline std::weak_ptr<logger_config_guard> g_active_guard;
inline std::mutex g_guard_mutex;

// Holds this library's configuration alive when it was set from outside, since the external
// caller has no object in this image to own it.
inline std::shared_ptr<logger_config_guard>& external_config_guard()
{
  // Force the logger's static to be constructed before this one, so it is destroyed after.
  // ~logger_config_guard calls reset_default_logger(), and at process exit an unpaired
  // guard released after the logger had already gone would touch a destroyed object.
  static rapids_logger::logger& keep_logger_alive = default_logger();
  static_cast<void>(keep_logger_alive);

  static std::shared_ptr<logger_config_guard> guard;
  return guard;
}

// Nesting depth of external configuration, so overlapping callers behave like overlapping
// init_logger_t instances: the outermost configuration wins and only its exit tears down.
inline int& external_config_depth()
{
  static int depth = 0;
  return depth;
}

/**
 * @brief Body of a component's exported configure entry point.
 *
 * Takes the same guard that init_logger_t takes, and keeps it alive. Library code that later
 * constructs an init_logger_t of its own -- the MIP and PDLP solve paths both do -- then sees
 * a live configuration and reuses it. Without that, the solver would reconfigure the logger
 * mid-run and, with truncate set, clear a log file the caller had already written to.
 */
inline void configure_logging_impl(const std::string& log_file, bool log_to_console, bool truncate)
{
  std::lock_guard<std::mutex> lock(g_guard_mutex);

  // An inner caller reuses the configuration already in place rather than replacing it,
  // matching init_logger_t. Reconfiguring here would also re-truncate the log file.
  if (external_config_depth()++ > 0) { return; }

  // Drop the previous guard *before* applying the new sinks. ~logger_config_guard calls
  // reset_default_logger(), so releasing it afterwards would run that reset on top of the
  // configuration we just applied and silently put the buffer sink back.
  external_config_guard().reset();

  try {
    apply_logger_config(log_file, log_to_console, truncate);
  } catch (...) {
    // Put the depth back. The caller's constructor is the one throwing, so its destructor
    // never runs to balance the increment, and a depth stuck above zero would make every
    // later configure look nested and silently do nothing -- logging dead for the process
    // because one log file could not be opened.
    --external_config_depth();
    reset_default_logger();
    throw;
  }

  auto guard              = std::make_shared<logger_config_guard>();
  g_active_guard          = guard;
  external_config_guard() = guard;
}

inline void reset_logging_impl()
{
  std::lock_guard<std::mutex> lock(g_guard_mutex);

  if (external_config_depth() == 0) { return; }
  if (--external_config_depth() > 0) { return; }

  external_config_guard().reset();
}

inline init_logger_t::init_logger_t(std::string log_file, bool log_to_console, bool truncate)
{
  std::lock_guard<std::mutex> lock(g_guard_mutex);

  auto existing_guard = g_active_guard.lock();
  if (existing_guard) {
    // Reuse existing configuration, just hold a reference to keep it alive
    guard_ = existing_guard;
    return;
  }

  apply_logger_config(log_file, log_to_console, truncate);

  // Create guard and store weak reference for future instances to find
  auto guard     = std::make_shared<logger_config_guard>();
  g_active_guard = guard;
  guard_         = guard;
}

/**
 * @brief Which component library's logger to configure.
 */
enum class log_target_t {
  mathopt,  ///< LP / MILP / QP, in cuopt_mathopt
  routing   ///< VRP, in cuopt_routing
};

}  // namespace cuopt

/*
 * Exported per-component entry points. Each is defined in exactly one component library and
 * configures that library's own hidden logger. They are the only logging symbols that cross
 * a library boundary.
 */
namespace cuopt::mathematical_optimization {
CUOPT_EXPORT void configure_logging(const std::string& log_file,
                                    bool log_to_console,
                                    bool truncate);
CUOPT_EXPORT void reset_logging();
}  // namespace cuopt::mathematical_optimization

#ifdef CUOPT_HAS_ROUTING
namespace cuopt::routing {
CUOPT_EXPORT void configure_logging(const std::string& log_file,
                                    bool log_to_console,
                                    bool truncate);
CUOPT_EXPORT void reset_logging();
}  // namespace cuopt::routing
#endif

namespace cuopt {

/**
 * @brief Configures a component library's logger from outside that library.
 *
 * `init_logger_t` configures the logger of whichever image constructs it, which is what
 * library code wants but not what an external caller wants: the CLI, the tests and the
 * Python bindings each hold their own logger and need to reach into the solver's. This
 * dispatches to the component's exported entry point instead.
 *
 * Defaults to mathopt because every external caller today is LP or MILP; routing is opted
 * into explicitly.
 */
class init_component_logger_t {
  log_target_t target_;

 public:
  explicit init_component_logger_t(const std::string& log_file,
                                   bool log_to_console,
                                   log_target_t target = log_target_t::mathopt,
                                   bool truncate       = true)
    : target_(target)
  {
    switch (target_) {
      case log_target_t::routing:
#ifdef CUOPT_HAS_ROUTING
        cuopt::routing::configure_logging(log_file, log_to_console, truncate);
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
        cuopt::mathematical_optimization::configure_logging(log_file, log_to_console, truncate);
        break;
    }
  }

  ~init_component_logger_t()
  {
    switch (target_) {
      case log_target_t::routing:
#ifdef CUOPT_HAS_ROUTING
        cuopt::routing::reset_logging();
#endif
        break;
      case log_target_t::mathopt:
      default: cuopt::mathematical_optimization::reset_logging(); break;
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
