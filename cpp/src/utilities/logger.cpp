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

// Guard object whose destructor resets the logger
struct logger_config_guard {
  ~logger_config_guard() { cuopt::reset_default_logger(); }
};

// Weak reference to detect if any init_logger_t instance is still alive
static std::weak_ptr<logger_config_guard> g_active_guard;
static std::mutex g_guard_mutex;

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

  // Create guard and store weak reference for future instances to find
  auto guard     = std::make_shared<logger_config_guard>();
  g_active_guard = guard;
  guard_         = guard;
}

}  // namespace cuopt
