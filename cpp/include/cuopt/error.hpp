/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "cuopt/linear_programming/constants.h"

#include <stdarg.h>

#include <raft/core/error.hpp>

namespace cuopt {

/**
 * @brief Indicates different type of exceptions which cuOpt might throw
 */
enum class error_type_t {
  Success          = CUOPT_SUCCESS,
  ValidationError  = CUOPT_VALIDATION_ERROR,
  OutOfMemoryError = CUOPT_OUT_OF_MEMORY,
  RuntimeError     = CUOPT_RUNTIME_ERROR
};

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUOPT_EXPECTS and  CUOPT_FAIL macros.
 *
 */

struct logic_error : public std::logic_error {
  explicit logic_error() = default;

  logic_error(const logic_error& exception) = default;

  // Move constructor
  logic_error(logic_error&& exception) = default;

  // Parameterized constructor
  logic_error(std::string const& message, error_type_t error_type)
    : std::logic_error(message), msg_(message), error_type_(error_type)
  {
  }

  // Assignment operators
  logic_error& operator=(const logic_error& exception) = default;

  logic_error& operator=(logic_error&& exception) = default;

  /**
   * @brief Returns the explanatory string.
   *
   * @return Pointer to a null-terminated string with explanatory information.
   */
  const char* what() const noexcept { return msg_.c_str(); }

  /**
   * @brief Returns the error type.
   *
   * @return The error type enum value.
   */
  error_type_t get_error_type() const noexcept { return error_type_; }

 private:
  std::string msg_;
  error_type_t error_type_;
};

/**
 * @brief Covert error enum type to string
 *
 * @param error error_type_t type enum value
 */
inline std::string error_to_string(error_type_t error)
{
  switch (error) {
    case error_type_t::Success: return std::string("Success");
    case error_type_t::ValidationError: return std::string("ValidationError");
    case error_type_t::RuntimeError: return std::string("RuntimeError");
    case error_type_t::OutOfMemoryError: return std::string("OutOfMemoryError");
  }

  return std::string("UnAccountedError");
}

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when a
 * condition is false
 *
 * @param[bool] cond From expression that evaluates to true or false
 * @param[error_type_t] error enum error type
 * @param[const char *] fmt String format for error message
 * @param variable set of arguments used for fmt
 * @throw cuopt::logic_error if the condition evaluates to false.
 */
inline void cuopt_expects(bool cond, error_type_t error_type, const char* fmt, ...)
{
  if (not cond) {
    va_list args;
    va_start(args, fmt);

    char msg[2048];
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    throw cuopt::logic_error("{\"CUOPT_ERROR_TYPE\": \"" + error_to_string(error_type) +
                               "\", \"msg\": " + "\"" + std::string(msg) + "\"}",
                             error_type);
  }
}

#define CUOPT_SET_ERROR_MSG(msg, location_prefix, fmt, ...)      \
  do {                                                           \
    char err_msg[2048]; /* NOLINT */                             \
    std::snprintf(err_msg, sizeof(err_msg), location_prefix);    \
    msg += err_msg;                                              \
    std::snprintf(err_msg, sizeof(err_msg), fmt, ##__VA_ARGS__); \
    msg += err_msg;                                              \
  } while (0)

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when a
 * condition is false
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] fmt String literal description of the reason that cond is expected
 * to be true with optinal format tagas
 * @throw cuopt::logic_error if the condition evaluates to false.
 */
#define EXE_CUOPT_EXPECTS(cond, fmt, ...)                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::string msg{};                                                       \
      CUOPT_SET_ERROR_MSG(msg, "NVIDIA cuOpt failure - ", fmt, ##__VA_ARGS__); \
      throw cuopt::logic_error(msg, error_type_t::RuntimeError);               \
    }                                                                          \
  } while (0)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * @param[in] fmt String literal description of the reason that this code path
 * is erroneous with optinal format tagas
 * @throw always throws cuopt::logic_error
 */
#define EXE_CUOPT_FAIL(fmt, ...)                                             \
  do {                                                                       \
    std::string msg{};                                                       \
    CUOPT_SET_ERROR_MSG(msg, "NVIDIA cuOpt failure - ", fmt, ##__VA_ARGS__); \
    throw cuopt::logic_error(msg, error_type_t::RuntimeError);               \
  } while (0)

/**
 * @brief function version of macro EXE_CUOPT_FAIL
 * This allow non literal type error messages
 *
 * @tparam Args
 * @param args
 */
template <typename... Args>
void execute_cuopt_fail(Args... args)
{
  auto msg = std::string("NVIDIA cuOpt failure - ");
  for (const auto& arg : {args...}) {
    msg += std::string(arg);
  }
  throw cuopt::logic_error(msg, error_type_t::RuntimeError);
}

}  // namespace cuopt
