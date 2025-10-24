/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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
#pragma once

#include <cuopt/logger.hpp>
#include <fstream>
#include <iostream>

namespace cuopt::linear_programming {
class init_logger_t {
 public:
  init_logger_t(std::string log_file, bool log_to_console = true)
  {
    // if (!log_to_console) {
    // popback the default sink
    cuopt::default_logger().sinks().pop_back();
    // }

    if (log_to_console) {
      cuopt::default_logger().sinks().push_back(
        std::make_shared<rapids_logger::ostream_sink_mt>(std::cout));
    }
    if (!log_file.empty()) {
      // TODO save the defaul sink and restore it
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
    auto buffered_messages = global_log_buffer().get();
    // std::cout << "Buffered messages: " << buffered_messages.size() << std::endl;
    for (const auto& msg : buffered_messages) {
      cuopt::default_logger().log(rapids_logger::level_enum::info, msg);
    }
    global_log_buffer().clear();
  }
  ~init_logger_t() { cuopt::reset_default_logger(); }
};
}  // namespace cuopt::linear_programming
