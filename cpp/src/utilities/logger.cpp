/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/logger.hpp>
#include <utilities/version_info.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>

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

log_buffer& global_log_buffer()
{
  static log_buffer buffer;
  return buffer;
}

// Callback function for the buffer sink
static void buffer_log_callback(int lvl, const char* msg)
{
  // store level with message; actual filtering happens at logger time
  global_log_buffer().log(static_cast<rapids_logger::level_enum>(lvl), msg);
}

/**
 * @brief Returns the default sink for the global logger.
 *
 * If the environment variable `CUOPT_DEBUG_LOG_FILE` is defined, the default sink is a sink to that
 * file. Otherwise, the default is to dump to stderr.
 *
 * @return sink_ptr The sink to use
 */
rapids_logger::sink_ptr default_sink()
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
 * @brief Runtime log-level override from the `CUOPT_LOG_LEVEL` environment variable.
 *
 * Accepts a level name (case-insensitive): TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, OFF.
 * Returns std::nullopt if the variable is unset or holds an unrecognised value.
 *
 * @note Statements below the compile-time `CUOPT_LOG_ACTIVE_LEVEL` (default INFO) are
 *  removed at build time, so raising verbosity above the build level has no effect;
 *  lowering it (e.g. WARN/ERROR/OFF to suppress output) always works.
 */
inline std::optional<rapids_logger::level_enum> env_log_level()
{
  const char* env = std::getenv("CUOPT_LOG_LEVEL");
  if (env == nullptr) { return std::nullopt; }
  std::string level{env};
  std::transform(level.begin(), level.end(), level.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  if (level == "TRACE") { return rapids_logger::level_enum::trace; }
  if (level == "DEBUG") { return rapids_logger::level_enum::debug; }
  if (level == "INFO") { return rapids_logger::level_enum::info; }
  if (level == "WARN") { return rapids_logger::level_enum::warn; }
  if (level == "ERROR") { return rapids_logger::level_enum::error; }
  if (level == "CRITICAL") { return rapids_logger::level_enum::critical; }
  if (level == "OFF") { return rapids_logger::level_enum::off; }
  return std::nullopt;  // unrecognised value: keep the compiled default
}

/**
 * @brief Returns the default log level for the global logger.
 *
 * The `CUOPT_LOG_LEVEL` environment variable, when set, overrides the compile-time default.
 *
 * @return rapids_logger::level_enum The default log level.
 */
inline rapids_logger::level_enum default_level()
{
  if (auto lvl = env_log_level()) { return *lvl; }
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

rapids_logger::logger& default_logger()
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

void reset_default_logger()
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

static std::mutex g_guard_mutex;

// Guard object whose destructor resets the logger.
struct logger_config_guard {
  ~logger_config_guard() { cuopt::reset_default_logger(); }
};

// Weak reference to detect if any init_logger_t instance is still alive
static std::weak_ptr<logger_config_guard> g_active_guard;

// Registration is per-thread, not global: the sink is shared by every solve, so
// the callback has to be selected by who is logging rather than by who
// registered last (#1752).
namespace {
struct thread_log_callback_t {
  log_callback_with_data_t callback = nullptr;
  void* user_data                   = nullptr;
};
thread_local thread_log_callback_t t_log_callback;
}  // namespace

static void user_log_bridge(int lvl, const char* msg)
{
  // Standard solver output only; debug/trace are internal diagnostics and must
  // not reach user code even in a lower-level build.
  if (lvl < static_cast<int>(rapids_logger::level_enum::info)) { return; }

  const auto& cb = t_log_callback;
  if (cb.callback) { cb.callback(msg, cb.user_data); }
}

log_callback_registration_t current_log_callback()
{
  return {t_log_callback.callback, t_log_callback.user_data};
}

scoped_log_callback_t::scoped_log_callback_t(log_callback_with_data_t cb, void* user_data)
  : prev_callback_(t_log_callback.callback), prev_user_data_(t_log_callback.user_data)
{
  t_log_callback.callback  = cb;
  t_log_callback.user_data = user_data;
}

scoped_log_callback_t::~scoped_log_callback_t()
{
  t_log_callback.callback  = prev_callback_;
  t_log_callback.user_data = prev_user_data_;
}

init_logger_t::init_logger_t(std::string log_file, bool log_to_console)
{
  std::lock_guard<std::mutex> lock(g_guard_mutex);

  auto existing_guard = g_active_guard.lock();
  if (existing_guard) {
    // Reuse existing configuration, just hold a reference to keep it alive.
    guard_ = existing_guard;
    return;
  }

  cuopt::default_logger().sinks().clear();

  // re-initialize sinks
  if (log_to_console) {
    cuopt::default_logger().sinks().push_back(
      std::make_shared<rapids_logger::ostream_sink_mt>(std::cout));
  }
  if (!log_file.empty()) {
    cuopt::default_logger().sinks().push_back(
      std::make_shared<rapids_logger::basic_file_sink_mt>(log_file, true));
    cuopt::default_logger().flush_on(rapids_logger::level_enum::debug);
  }
  auto guard = std::make_shared<logger_config_guard>();

  // Always installed: the bridge no-ops unless the logging thread has a
  // registration, so delivery no longer depends on which solve built the guard.
  cuopt::default_logger().sinks().push_back(
    std::make_shared<rapids_logger::callback_sink_mt>(user_log_bridge));

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

  g_active_guard = guard;
  guard_         = guard;
}

}  // namespace cuopt
