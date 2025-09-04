/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <string>

#include <stdarg.h>
#include <stdexcept>

namespace cuopt::mps_parser {

/**
 * @brief Indicates different type of exceptions which mps parser might throw
 */
enum class error_type_t { ValidationError, OutOfMemoryError, RuntimeError };

/**
 * @brief Covert error enum type to string
 *
 * @param error error_type_t type enum value
 */
inline std::string error_to_string(error_type_t error)
{
  switch (error) {
    case error_type_t::ValidationError: return std::string("ValidationError");
    case error_type_t::RuntimeError: return std::string("RuntimeError");
    case error_type_t::OutOfMemoryError: return std::string("OutOfMemoryError");
  }

  return std::string("UnAccountedError");
}

/**
 * @brief Function for checking (pre-)conditions that throws an exception when a
 * condition is false
 *
 * @param[bool] cond From expression that evaluates to true or false
 * @param[error_type_t] error enum error type
 * @param[const char *] fmt String format for error message
 * @param variable set of arguments used for fmt
 * @throw std::logic_error if the condition evaluates to false.
 */
inline void mps_parser_expects(bool cond, error_type_t error_type, const char* fmt, ...)
{
  if (not cond) {
    va_list args;
    va_start(args, fmt);

    char msg[2048];
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    throw std::logic_error("{\"MPS_PARSER_ERROR_TYPE\": \"" + error_to_string(error_type) +
                           "\", \"msg\": " + "\"" + std::string(msg) + "\"}");
  }
}

/**
 * @brief Function for checking (pre-)conditions that aborts the program when a
 * condition is false
 *
 * @param[bool] cond From expression that evaluates to true or false
 * @param[error_type_t] error enum error type
 * @param[const char *] fmt String format for error message
 * @param variable set of arguments used for fmt
 * @throw std::logic_error if the condition evaluates to false.
 */
inline void mps_parser_expects_fatal(bool cond, error_type_t error_type, const char* fmt, ...)
{
  if (not cond) {
    va_list args;
    va_start(args, fmt);

    char msg[2048];
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
    std::string error_string = error_to_string(error_type);
    std::fprintf(stderr,
                 "{\"MPS_PARSER_ERROR_TYPE\": \"%s\", \"msg\": \"%s\"}\n",
                 error_to_string(error_type).c_str(),
                 msg);
    std::fflush(stderr);
    std::abort();
  }
}

#define MPS_PARSER_SET_ERROR_MSG(msg, location_prefix, fmt, ...) \
  do {                                                           \
    char err_msg[2048]; /* NOLINT */                             \
    std::snprintf(err_msg, sizeof(err_msg), location_prefix);    \
    msg += err_msg;                                              \
    std::snprintf(err_msg, sizeof(err_msg), fmt, ##__VA_ARGS__); \
    msg += err_msg;                                              \
  } while (0)

/**
 * @brief Macro for checking if an exception is thrown that throws an exception when statement
 * throws an exception
 *
 * @param[in] statement Statement that might throw an exception
 * @param[in] error_type Error type to be thrown
 * @param[in] fmt String literal description of the reason that cond is expected
 * to be true with optinal format tagas
 */
#define mps_parser_no_except(statement, error_type, fmt, ...)                                 \
  do {                                                                                        \
    try {                                                                                     \
      statement                                                                               \
    } catch (...) {                                                                           \
      std::string msg{};                                                                      \
      MPS_PARSER_SET_ERROR_MSG(msg, "NVIDIA mps parser failure - ", fmt, ##__VA_ARGS__);      \
      throw std::logic_error("{\"MPS_PARSER_ERROR_TYPE\": \"" + error_to_string(error_type) + \
                             "\", \"msg\": " + "\"" + msg + "\"}");                           \
    }                                                                                         \
  } while (0)

}  // namespace cuopt::mps_parser
