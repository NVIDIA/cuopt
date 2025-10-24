/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuopt/logger.hpp>

#include <utilities/version_info.hpp>
namespace cuopt {

void log_buffer::add(const char* msg)
{
  std::lock_guard<std::mutex> lock(mutex);
  std::string str(msg);

  if (!str.empty() && str.back() == '\n') { str.pop_back(); }
  messages.emplace_back(str);
}

std::vector<std::string> log_buffer::get() const
{
  std::lock_guard<std::mutex> lock(mutex);
  return messages;
}

void log_buffer::clear()
{
  std::lock_guard<std::mutex> lock(mutex);
  messages.clear();
}

log_buffer& global_log_buffer()
{
  static log_buffer buffer;
  return buffer;
}

// Callback function for the buffer sink
static void buffer_log_callback(int lvl, const char* msg)
{
  // FIXME:: check for levels?
  global_log_buffer().add(msg);
}

rapids_logger::sink_ptr default_sink()
{
  return std::make_shared<rapids_logger::callback_sink_mt>(buffer_log_callback);
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

}  // namespace cuopt
