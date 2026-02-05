/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

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
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

// Problem data for the mixed integer linear programming problem
mps_parser::mps_data_model_t<int, double> create_cuts_problem_1()
{
  // Create problem instance
  mps_parser::mps_data_model_t<int, double> problem;

  // Solve the problem
  // minimize -7*x1 -2*x2
  // subject to -1*x1 + 2*x2 <= 4
  //            5*x1 + 1*x2 <= 20
  //            -2*x1 -2*x2 <= -7

  // Set up constraint matrix in CSR format
  std::vector<int> offsets         = {0, 2, 4, 6};
  std::vector<int> indices         = {0, 1, 0, 1, 0, 1};
  std::vector<double> coefficients = {-1.0, 2.0, 5.0, 1.0, -2.0, -2.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());

  // Set constraint bounds
  std::vector<double> lower_bounds = {-std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity()};
  std::vector<double> upper_bounds = {4.0, 20.0, -7.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());

  // Set variable bounds
  std::vector<double> var_lower_bounds = {0.0, 0.0};
  std::vector<double> var_upper_bounds = {10.0, 10.0};
  problem.set_variable_lower_bounds(var_lower_bounds.data(), var_lower_bounds.size());
  problem.set_variable_upper_bounds(var_upper_bounds.data(), var_upper_bounds.size());

  // Set objective coefficients (minimize -7*x1 -2*x2)
  std::vector<double> objective_coefficients = {-7.0, -2.0};
  problem.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());

  // Set variable types
  std::vector<char> variable_types = {'I', 'I'};
  problem.set_variable_types(variable_types);

  return problem;
}

TEST(cuts, test_cuts_1)
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  constexpr double test_time_limit = 1.;

  // Create the problem
  auto problem = create_cuts_problem_1();

  settings.time_limit                  = test_time_limit;
  settings.max_cut_passes              = 1;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);

  double obj_val = solution.get_objective_value();
  // Expected objective value from documentation example is approximately -28
  EXPECT_NEAR(-28, obj_val, 1e-3);

  EXPECT_EQ(solution.get_num_nodes(), 0);
}

// Problem data for the mixed integer linear programming problem
mps_parser::mps_data_model_t<int, double> create_cuts_problem_2()
{
  // Create problem instance
  mps_parser::mps_data_model_t<int, double> problem;

  // Solve the problem
  // minimize -86*y1 -4*y2 -40*y3
  // subject to 774*y1 + 76*y2 + 42*y3 <= 875
  //            67*y1 + 27*y2 + 53*y3 <= 875
  //            y1, y2, y3 in {0, 1}

  // Set up constraint matrix in CSR format
  std::vector<int> offsets         = {0, 3, 6};
  std::vector<int> indices         = {0, 1, 2, 0, 1, 2};
  std::vector<double> coefficients = {774.0, 76.0, 42.0, 67.0, 27.0, 53.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());

  // Set constraint bounds
  std::vector<double> lower_bounds = {-std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity()};
  std::vector<double> upper_bounds = {875.0, 875.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());

  // Set variable bounds
  std::vector<double> var_lower_bounds = {0.0, 0.0, 0.0};
  std::vector<double> var_upper_bounds = {1.0, 1.0, 1.0};
  problem.set_variable_lower_bounds(var_lower_bounds.data(), var_lower_bounds.size());
  problem.set_variable_upper_bounds(var_upper_bounds.data(), var_upper_bounds.size());

  // Set objective coefficients (minimize -86*y1 -4*y2 -40*y3)
  std::vector<double> objective_coefficients = {-86.0, -4.0, -40.0};
  problem.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());

  // Set variable types
  std::vector<char> variable_types = {'I', 'I', 'I'};
  problem.set_variable_types(variable_types);

  return problem;
}

TEST(cuts, test_cuts_2)
{
  const raft::handle_t handle_{};
  mip_solver_settings_t<int, double> settings;
  constexpr double test_time_limit = 1.;

  // Create the problem
  auto problem = create_cuts_problem_2();

  settings.time_limit                  = test_time_limit;
  settings.max_cut_passes              = 10;
  settings.presolver                   = presolver_t::None;
  mip_solution_t<int, double> solution = solve_mip(&handle_, problem, settings);
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);

  double obj_val = solution.get_objective_value();
  // Expected objective value from documentation example is approximately -126
  EXPECT_NEAR(-126, obj_val, 1e-3);

  EXPECT_EQ(solution.get_num_nodes(), 0);
}

}  // namespace cuopt::linear_programming::test
