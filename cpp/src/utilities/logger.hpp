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

class init_logger_t {
 public:
  init_logger_t(std::string log_file, bool log_to_console);

  ~init_logger_t();
};

}  // namespace cuopt
