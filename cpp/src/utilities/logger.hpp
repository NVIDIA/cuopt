/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/logger_macros.hpp>

#include <rapids_logger/logger.hpp>

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cuopt {

/**
 * @brief Get the default logger.
 *
 * @return logger& The default logger
 */
rapids_logger::logger& default_logger();

/**
 * @brief Reset the default logger to the default settings.
 *  This is needed when we are running multiple tests and each test has different logger settings
 *  and we need to reset the logger to the default settings before each test.
 */
void reset_default_logger();

// Ref-counted logger initializer
class init_logger_t {
  // Using shared_ptr for ref-counting
  std::shared_ptr<void> guard_;

 public:
  init_logger_t(std::string log_file, bool log_to_console);
};

}  // namespace cuopt
