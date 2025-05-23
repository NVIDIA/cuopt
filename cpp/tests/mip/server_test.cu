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
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

// Create standard LP test problem matching Python test
mps_parser::mps_data_model_t<int, double> create_std_lp_problem()
{
  mps_parser::mps_data_model_t<int, double> problem;

  // Set up constraint matrix in CSR format
  std::vector<int> offsets         = {0, 2};
  std::vector<int> indices         = {0, 1};
  std::vector<double> coefficients = {1.0, 1.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());

  // Set constraint bounds
  std::vector<double> lower_bounds = {0.0};
  std::vector<double> upper_bounds = {5000.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());

  // Set variable bounds
  std::vector<double> var_lower = {0.0, 0.0};
  std::vector<double> var_upper = {3000.0, 5000.0};
  problem.set_variable_lower_bounds(var_lower.data(), var_lower.size());
  problem.set_variable_upper_bounds(var_upper.data(), var_upper.size());

  // Set objective coefficients
  std::vector<double> obj_coeffs = {1.2, 1.7};
  problem.set_objective_coefficients(obj_coeffs.data(), obj_coeffs.size());
  problem.set_maximize(false);

  return problem;
}

// Create standard MILP test problem matching Python test
mps_parser::mps_data_model_t<int, double> create_std_milp_problem(bool maximize)
{
  auto problem = create_std_lp_problem();

  // Set variable types for MILP
  std::vector<char> var_types = {'I', 'C'};
  problem.set_variable_types(var_types);
  problem.set_maximize(maximize);

  return problem;
}

TEST(ServerTest, TestSampleLP)
{
  raft::handle_t handle;
  auto problem = create_std_lp_problem();

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings{};
  settings.set_optimality_tolerance(1e-4);
  settings.set_time_limit(5);

  auto result = cuopt::linear_programming::solve_lp(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(),
            cuopt::linear_programming::pdlp_termination_status_t::Optimal);
}

class MILPTestParams
  : public testing::TestWithParam<
      std::tuple<bool, bool, bool, cuopt::linear_programming::mip_termination_status_t>> {};

TEST_P(MILPTestParams, TestSampleMILP)
{
  bool maximize                    = std::get<0>(GetParam());
  bool scaling                     = std::get<1>(GetParam());
  bool heuristics_only             = std::get<2>(GetParam());
  auto expected_termination_status = std::get<3>(GetParam());

  raft::handle_t handle;
  auto problem = create_std_milp_problem(maximize);

  cuopt::linear_programming::mip_solver_settings_t<int, double> settings{};
  settings.set_time_limit(5);
  settings.set_mip_scaling(scaling);
  settings.set_heuristics_only(heuristics_only);

  auto result = cuopt::linear_programming::solve_mip(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(), expected_termination_status);
}

INSTANTIATE_TEST_SUITE_P(
  MILPTests,
  MILPTestParams,
  testing::Values(
    std::make_tuple(
      true, true, true, cuopt::linear_programming::mip_termination_status_t::FeasibleFound),
    std::make_tuple(
      false, true, false, cuopt::linear_programming::mip_termination_status_t::Optimal),
    std::make_tuple(
      true, false, true, cuopt::linear_programming::mip_termination_status_t::FeasibleFound),
    std::make_tuple(
      false, false, false, cuopt::linear_programming::mip_termination_status_t::Optimal)));

}  // namespace cuopt::linear_programming::test
