/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

// Problem data for the mixed integer linear programming example from documentation
mps_parser::mps_data_model_t<int, double> create_doc_example_problem()
{
  // Create problem instance
  mps_parser::mps_data_model_t<int, double> problem;

  // Set up constraint matrix in CSR format
  std::vector<int> offsets         = {0, 2, 4};
  std::vector<int> indices         = {0, 1, 0, 1};
  std::vector<double> coefficients = {2.0, 4.0, 3.0, 2.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());

  // Set constraint bounds
  std::vector<double> lower_bounds = {230.0, -std::numeric_limits<double>::infinity()};
  std::vector<double> upper_bounds = {std::numeric_limits<double>::infinity(), 190.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());

  // Set variable bounds
  std::vector<double> var_lower_bounds = {0.0, 0.0};
  std::vector<double> var_upper_bounds = {std::numeric_limits<double>::infinity(),
                                          std::numeric_limits<double>::infinity()};
  problem.set_variable_lower_bounds(var_lower_bounds.data(), var_lower_bounds.size());
  problem.set_variable_upper_bounds(var_upper_bounds.data(), var_upper_bounds.size());

  // Set objective coefficients (maximize 5x + 3y)
  std::vector<double> objective_coefficients = {5.0, 3.0};
  problem.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());

  // Set variable types (x is integer, y is continuous)
  std::vector<char> variable_types = {'I', 'C'};  // 'I' for Integer, 'C' for Continuous
  problem.set_variable_types(variable_types);

  // Set to maximize
  problem.set_maximize(true);

  // Optional: Set variable names for better debugging
  problem.set_variable_names({"x", "y"});

  return problem;
}

struct result_map_t {
  std::string file;
  double cost;
};

void test_mps_file()
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  constexpr double test_time_limit = 1.;

  // Create the problem from documentation example
  auto problem = create_doc_example_problem();

  settings.time_limit                  = test_time_limit;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);

  double obj_val = solution.get_objective_value();
  // Expected objective value from documentation example is approximately 303.5
  EXPECT_NEAR(303.5, obj_val, 1.0);

  // Test solution bounds
  test_variable_bounds(problem, solution.get_solution(), settings);

  // Get solution values
  const auto& sol_values = solution.get_solution();
  // x should be approximately 37 and integer
  EXPECT_NEAR(37.0, sol_values.element(0, handle_.get_stream()), 0.1);
  EXPECT_NEAR(std::round(sol_values.element(0, handle_.get_stream())),
              sol_values.element(0, handle_.get_stream()),
              settings.tolerances.integrality_tolerance);  // Check x is integer
  // y should be approximately 39.5
  EXPECT_NEAR(39.5, sol_values.element(1, handle_.get_stream()), 0.1);
}

TEST(docs, mixed_integer_linear_programming) { test_mps_file(); }

TEST(docs, user_problem_file)
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  constexpr double test_time_limit = 1.;

  // Create the problem from documentation example
  auto problem = create_doc_example_problem();

  EXPECT_FALSE(std::filesystem::exists("user_problem.mps"));

  settings.time_limit        = test_time_limit;
  settings.user_problem_file = "user_problem.mps";
  EXPECT_EQ(solve_mip(&handle_, problem, settings).get_termination_status(),
            mip_termination_status_t::Optimal);

  EXPECT_TRUE(std::filesystem::exists("user_problem.mps"));

  cuopt::mps_parser::mps_data_model_t<int, double> problem2 =
    cuopt::mps_parser::parse_mps<int, double>("user_problem.mps", false);

  EXPECT_EQ(problem2.get_n_variables(), problem.get_n_variables());
  EXPECT_EQ(problem2.get_n_constraints(), problem.get_n_constraints());
  EXPECT_EQ(problem2.get_nnz(), problem.get_nnz());

  settings.user_problem_file           = "user_problem2.mps";
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem2, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);

  double obj_val = solution.get_objective_value();
  // Expected objective value from documentation example is approximately 303.5
  EXPECT_NEAR(303.5, obj_val, 1.0);

  // Get solution values
  const auto& sol_values = solution.get_solution();
  // x should be approximately 37 and integer
  EXPECT_NEAR(37.0, sol_values.element(0, handle_.get_stream()), 0.1);
  EXPECT_NEAR(std::round(sol_values.element(0, handle_.get_stream())),
              sol_values.element(0, handle_.get_stream()),
              settings.tolerances.integrality_tolerance);  // Check x is integer
  // y should be approximately 39.5
  EXPECT_NEAR(39.5, sol_values.element(1, handle_.get_stream()), 0.1);
}

}  // namespace cuopt::linear_programming::test
