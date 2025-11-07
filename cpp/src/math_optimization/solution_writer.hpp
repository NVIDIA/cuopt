/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

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
