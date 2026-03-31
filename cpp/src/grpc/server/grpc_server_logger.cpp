/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_server_logger.hpp"

#include <rapids_logger/logger.hpp>

#include <iostream>

static rapids_logger::logger create_default_server_logger()
{
  rapids_logger::logger logger{"CUOPT_SERVER",
                               {std::make_shared<rapids_logger::ostream_sink_mt>(std::cout)}};
  logger.set_pattern("[%Y-%m-%d %H:%M:%S.%e] %v");
  logger.set_level(rapids_logger::level_enum::info);
  logger.flush_on(rapids_logger::level_enum::info);
  return logger;
}

rapids_logger::logger& server_logger()
{
  static rapids_logger::logger logger_ = create_default_server_logger();
  return logger_;
}

void init_server_logger(const std::string& log_file, bool to_console, bool verbose)
{
  server_logger().sinks().clear();

  if (to_console) {
    server_logger().sinks().push_back(std::make_shared<rapids_logger::ostream_sink_mt>(std::cout));
  }
  if (!log_file.empty()) {
    server_logger().sinks().push_back(
      std::make_shared<rapids_logger::basic_file_sink_mt>(log_file, false));
  }

  server_logger().set_pattern("[%Y-%m-%d %H:%M:%S.%e] %v");
  auto level = verbose ? rapids_logger::level_enum::debug : rapids_logger::level_enum::info;
  server_logger().set_level(level);
  server_logger().flush_on(level);
}

#endif  // CUOPT_ENABLE_GRPC
