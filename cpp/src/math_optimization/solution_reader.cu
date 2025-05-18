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

#include "solution_reader.hpp"

#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
namespace cuopt::linear_programming {

/**
 * @brief Represents information about a solution including variables, objective value, and status
 *
 * This struct stores solution information read from a .sol file, including:
 * - Variable name to value mappings
 * - Optional objective value
 * - Optional solution status string
 */
struct solution_info_t {
  std::unordered_map<std::string, double> variables;
  std::optional<double> objective_value;
  std::optional<std::string> status;
};

/**
 * @brief Reads a solution file and parses its contents
 *
 * @param filename Path to the .sol file to read
 * @return solution_info_t Struct containing the parsed solution information
 * @throws std::runtime_error if the file cannot be opened
 *
 * This function reads a .sol file and extracts:
 * - Variable assignments in format "varname value"
 * - Objective value line starting with "# Objective value = "
 * - Status line starting with "# Status: "
 *
 * Lines not matching these patterns are ignored. Comments starting with # that don't match
 * the objective/status patterns are skipped.
 */

solution_info_t read_sol_file(const std::string& filename)
{
  solution_info_t info;
  std::ifstream file(filename);

  if (!file.is_open()) { throw std::runtime_error("Could not open file: " + filename); }

  std::string line;
  while (std::getline(file, line)) {
    // Skip empty lines
    if (line.empty()) { continue; }

    // Skip comments that don't contain special info
    if (line[0] == '#' || line[0] == '=') {
      std::string lower_line = line;
      std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);

      // Check for objective value
      if (size_t pos = lower_line.find("objective value"); pos != std::string::npos) {
        size_t value_start = line.find_first_of("-+.0123456789", pos);
        if (value_start != std::string::npos) {
          info.objective_value = std::stod(line.substr(value_start));
        }
        continue;
      }

      // Check for MIPLIB format as well
      if (size_t pos = lower_line.find("obj"); pos != std::string::npos) {
        size_t value_start = line.find_first_of("-+.0123456789", pos);
        if (value_start != std::string::npos) {
          info.objective_value = std::stod(line.substr(value_start));
        }
        continue;
      }

      // Check for status
      if (size_t pos = lower_line.find("status:"); pos != std::string::npos) {
        size_t status_start = line.find_first_not_of(" \t:", pos + 7);
        if (status_start != std::string::npos) {
          size_t status_end = line.find_first_of(" \t\n", status_start);
          info.status       = line.substr(status_start, status_end - status_start);
        }
        continue;
      }

      continue;
    }

    // Parse variable assignments
    std::istringstream iss(line);
    std::string var_name;
    double value;

    if (iss >> var_name >> value) { info.variables[var_name] = value; }
  }

  return info;
}

// Helper method to get a specific variable value
static double get_variable_value(const solution_info_t& info, const std::string& variable_name)
{
  auto it = info.variables.find(variable_name);
  if (it == info.variables.end()) {
    throw std::runtime_error("Variable not found in solution: " + variable_name);
  }
  return it->second;
}

/**
 * @brief Reads a solution file and returns the values of specified variables
 *
 * @param sol_file_path Path to the .sol file to read
 * @param variable_names Vector of variable names to extract values for
 * @return std::vector<double> Vector of values corresponding to the variable names
 */
std::vector<double> solution_reader_t::get_variable_values_from_sol_file(
  const std::string& sol_file_path, const std::vector<std::string>& variable_names)
{
  auto info = read_sol_file(sol_file_path);
  std::vector<double> values;
  for (const auto& var_name : variable_names) {
    values.push_back(get_variable_value(info, var_name));
  }
  return values;
}

}  // namespace cuopt::linear_programming
