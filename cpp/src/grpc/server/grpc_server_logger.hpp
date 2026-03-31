/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUOPT_ENABLE_GRPC

#include <rapids_logger/logger.hpp>
#include <string>

// Independent logger for gRPC server operational messages (startup, job
// lifecycle, throughput, IPC).  Completely separate from the solver logger
// (cuopt::default_logger / CUOPT_LOG_INFO) so the two never interfere.
//
// Created before fork() — both main and worker processes share the same
// stdout/file descriptors, so all output goes to one place.
rapids_logger::logger& server_logger();

// Reconfigure server logger sinks and level.  Call once in main() after
// argument parsing, before fork().
void init_server_logger(const std::string& log_file = {},
                        bool to_console             = true,
                        bool verbose                = true);

// Convenience macros — use these instead of std::cout throughout the server.
// rapids_logger uses printf-style format specifiers (%d, %s, %ld, etc.).
// Unconditional operational messages use INFO; verbose/diagnostic messages
// use DEBUG (suppressed when --quiet is passed).
#define SERVER_LOG_INFO(...)  server_logger().info(__VA_ARGS__)
#define SERVER_LOG_DEBUG(...) server_logger().debug(__VA_ARGS__)
#define SERVER_LOG_WARN(...)  server_logger().warn(__VA_ARGS__)
#define SERVER_LOG_ERROR(...) server_logger().error(__VA_ARGS__)

#endif  // CUOPT_ENABLE_GRPC
