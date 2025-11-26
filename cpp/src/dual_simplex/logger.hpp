/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#ifdef CUOPT_LOG_ACTIVE_LEVEL
#include <utilities/logger.hpp>
#endif

#include <string>

#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace cuopt::linear_programming::dual_simplex {

class logger_t {
 public:
  logger_t()
    : log(true),
      log_to_console(true),
      log_to_file(false),
      log_filename("dual_simplex.log"),
      log_file(nullptr)
  {
  }

  void enable_log_to_file(std::string mode = "w")
  {
    if (log_file != nullptr) { std::fclose(log_file); }
    log_file    = std::fopen(log_filename.c_str(), mode.c_str());
    log_to_file = true;
  }

  void set_log_file(const std::string& filename)
  {
    log_filename = filename;
    enable_log_to_file();
  }

  void close_log_file()
  {
    if (log_file != nullptr) { std::fclose(log_file); }
  }

  void printf(const char* fmt, ...)
  {
    if (log) {
#ifdef CUOPT_LOG_ACTIVE_LEVEL
      if (log_to_console) {
        char buffer[1024];
        std::va_list args;
        va_start(args, fmt);
        std::vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);

        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') { buffer[len - 1] = '\0'; }
        CUOPT_LOG_INFO("%s%s", log_prefix.c_str(), buffer);
      }
#else
      if (log_to_console) {
        std::va_list args;
        va_start(args, fmt);
        std::vprintf(fmt, args);
        va_end(args);
        fflush(stdout);
      }
#endif
      if (log_to_file && log_file != nullptr) {
        std::va_list args;
        va_start(args, fmt);
        std::vfprintf(log_file, fmt, args);
        va_end(args);
        fflush(log_file);
      }
    }
  }

  void debug([[maybe_unused]] const char* fmt, ...)
  {
    if (log) {
#ifdef CUOPT_LOG_DEBUG
      if (log_to_console) {
        char buffer[1024];
        std::va_list args;
        va_start(args, fmt);
        std::vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);

        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') { buffer[len - 1] = '\0'; }
        CUOPT_LOG_TRACE("%s%s", log_prefix.c_str(), buffer);
      }
#else
      if (log_to_console) {
        std::va_list args;
        va_start(args, fmt);
        std::vprintf(fmt, args);
        va_end(args);
        fflush(stdout);
      }
#endif
      if (log_to_file && log_file != nullptr) {
        std::va_list args;
        va_start(args, fmt);
        std::vfprintf(log_file, fmt, args);
        va_end(args);
        fflush(log_file);
      }
    }
  }

  bool log;
  bool log_to_console;
  std::string log_prefix;

 private:
  bool log_to_file;
  std::string log_filename;
  std::FILE* log_file;
};

}  // namespace cuopt::linear_programming::dual_simplex
