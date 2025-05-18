/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <vector>

namespace cuopt::linear_programming {

/**
 * @brief Reads a solution file and returns the values of specified variables
 *
 * @param sol_file_path Path to the .sol file to read
 * @param variable_names Vector of variable names to extract values for
 * @return std::vector<double> Vector of values corresponding to the variable names
 */
class solution_reader_t {
 public:
  static std::vector<double> get_variable_values_from_sol_file(
    const std::string& sol_file_path, const std::vector<std::string>& variable_names);
};
}  // namespace cuopt::linear_programming