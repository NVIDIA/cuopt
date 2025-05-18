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
 * @brief Writes a solution to a .sol file
 *
 * @param sol_file_path Path to the .sol file to write
 * @param status Status of the solution
 * @param objective_value Objective value of the solution
 * @param variable_names Vector of variable names
 * @param variable_values Vector of variable values
 */
class solution_writer_t {
 public:
  static void write_solution_to_sol_file(const std::string& sol_file_path,
                                         const std::string& status,
                                         const double objective_value,
                                         const std::vector<std::string>& variable_names,
                                         const std::vector<double>& variable_values);
};
}  // namespace cuopt::linear_programming