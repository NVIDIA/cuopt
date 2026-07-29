/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/logger.hpp>
#include <utilities/version_info.hpp>

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

// Forward declarations needed by logger_config_guard destructor.
static std::mutex g_guard_mutex;
static const struct captured_log_callback_t* g_active_log_callback;

// Captured (immutable) callback state owned by the active logger guard.
struct captured_log_callback_t {
  log_callback_with_data_t callback;
  void* user_data;
};

// Guard object whose destructor resets the logger.
// Owns the captured callback state to guarantee its lifetime.
struct logger_config_guard {
  std::unique_ptr<captured_log_callback_t> callback_state;
  ~logger_config_guard()
  {
    cuopt::reset_default_logger();  // removes the sink; blocks until in-flight log calls finish
    std::lock_guard<std::mutex> lock(g_guard_mutex);
    g_active_log_callback = nullptr;  // safe: the sink (and the bridge) are already gone
  }
};

// Weak reference to detect if any init_logger_t instance is still alive
static std::weak_ptr<logger_config_guard> g_active_guard;

// g_active_log_callback: written only under g_guard_mutex (at guard create/destroy time).
// Read lock-free by user_log_bridge — safe because the bridge is only reachable
// while the sink is alive, and the sink is removed (in reset_default_logger) before
// this pointer is cleared.

// Pending user log callback/level set by the C API before cuOptSolve.
// Consumed once (under g_guard_mutex) by init_logger_t to build the guard state.
static log_callback_with_data_t g_pending_callback = nullptr;
static void* g_pending_callback_data               = nullptr;
static int g_pending_log_level                     = -1;  // -1 = use compiled default

static void user_log_bridge(int lvl, const char* msg)
{
  // g_active_log_callback is stable for the duration of any bridge call:
  // it points into the guard's callback_state, which outlives the sink.
  const captured_log_callback_t* state = g_active_log_callback;
  if (state) { state->callback(lvl, msg, state->user_data); }
}

void set_pending_log_callback(log_callback_with_data_t cb, void* user_data)
{
  std::lock_guard<std::mutex> lock(g_guard_mutex);
  g_pending_callback      = cb;
  g_pending_callback_data = user_data;
}

void clear_pending_log_callback()
{
  std::lock_guard<std::mutex> lock(g_guard_mutex);
  g_pending_callback      = nullptr;
  g_pending_callback_data = nullptr;
}

void set_pending_log_level(int level)
{
  std::lock_guard<std::mutex> lock(g_guard_mutex);
  g_pending_log_level = level;
}

void clear_pending_log_level()
{
  std::lock_guard<std::mutex> lock(g_guard_mutex);
  g_pending_log_level = -1;
}

init_logger_t::init_logger_t(std::string log_file, bool log_to_console)
{
  std::lock_guard<std::mutex> lock(g_guard_mutex);

  auto existing_guard = g_active_guard.lock();
  if (existing_guard) {
    // Reuse existing configuration, just hold a reference to keep it alive
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
  // Capture pending callback into the guard so the bridge reads stable (immutable) state.
  auto guard = std::make_shared<logger_config_guard>();
  if (g_pending_callback) {
    guard->callback_state =
      std::make_unique<captured_log_callback_t>(captured_log_callback_t{g_pending_callback, g_pending_callback_data});
    g_active_log_callback = guard->callback_state.get();
    cuopt::default_logger().sinks().push_back(
      std::make_shared<rapids_logger::callback_sink_mt>(user_log_bridge));
  }

#if CUOPT_LOG_ACTIVE_LEVEL >= RAPIDS_LOGGER_LOG_LEVEL_INFO
  cuopt::default_logger().set_pattern("%v");
#else
  cuopt::default_logger().set_pattern(cuopt::default_pattern());
#endif

  if (g_pending_log_level >= 0) {
    cuopt::default_logger().set_level(static_cast<rapids_logger::level_enum>(g_pending_log_level));
  }

  // Extract messages from the global buffer and log to the default logger
  auto buffered_messages = global_log_buffer().drain_all();
  for (const auto& entry : buffered_messages) {
    cuopt::default_logger().log(entry.level, entry.msg.c_str());
  }

  g_active_guard = guard;
  guard_         = guard;
}

}  // namespace cuopt
