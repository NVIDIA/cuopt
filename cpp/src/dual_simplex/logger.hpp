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

  void enable_log_to_file()
  {
    if (log_file != nullptr) { std::fclose(log_file); }
    log_file    = std::fopen(log_filename.c_str(), "w");
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
        CUOPT_LOG_INFO(buffer);
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
        CUOPT_LOG_TRACE(buffer);
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

 private:
  bool log_to_file;
  std::string log_filename;
  std::FILE* log_file;
};

}  // namespace cuopt::linear_programming::dual_simplex
