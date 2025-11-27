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
