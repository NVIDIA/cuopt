/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

#include <linear_programming/utils.cuh>
#include <mps_parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace cuopt::linear_programming::test {
constexpr double tolerance = 1e-6f;

std::string make_path_absolute(const std::string& file)
{
  std::string rel_file{};
  // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  rel_file                                = rapidsDatasetRootDir + "/" + file;
  return rel_file;
}

// Compute on the CPU x * c to check that the returned objective value is correct
static void test_objective_sanity(
  const cuopt::mps_parser::mps_data_model_t<int, double>& op_problem,
  const rmm::device_uvector<double>& primal_solution,
  double objective_value,
  double epsilon = tolerance)
{
  const auto primal_vars = host_copy(primal_solution);
  const auto& c_vector   = op_problem.get_objective_coefficients();
  std::vector<double> out(primal_vars.size());
  std::transform(primal_vars.cbegin(),
                 primal_vars.cend(),
                 c_vector.cbegin(),
                 out.begin(),
                 std::multiplies<double>());

  double sum = std::reduce(out.cbegin(), out.cend(), 0.0);

  EXPECT_NEAR(sum, objective_value, epsilon);
}

// Compute A @ x, compute the residual (distance to combined bounds)
//  Check that it corresponds to the bound resdiual
//  Check that it respect the absolute/relative tolerance
// Check that the primal variables respected the variable bounds
static void test_constraint_sanity(
  const cuopt::mps_parser::mps_data_model_t<int, double>& op_problem,
  const optimization_problem_solution_t<int, double>& solution,
  double epsilon = tolerance)
{
  const std::vector<double> primal_vars              = host_copy(solution.get_primal_solution());
  const std::vector<double>& values                  = op_problem.get_constraint_matrix_values();
  const std::vector<int>& indices                    = op_problem.get_constraint_matrix_indices();
  const std::vector<int>& offsets                    = op_problem.get_constraint_matrix_offsets();
  const std::vector<double>& constraint_lower_bounds = op_problem.get_constraint_lower_bounds();
  const std::vector<double>& constraint_upper_bounds = op_problem.get_constraint_upper_bounds();
  const std::vector<double>& variable_lower_bounds   = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds   = op_problem.get_variable_upper_bounds();
  std::vector<double> residual(solution.get_dual_solution().size(), 0.0);
  std::vector<double> viol(solution.get_dual_solution().size(), 0.0);

  // CSR SpMV
  for (size_t i = 0; i < offsets.size() - 1; ++i) {
    for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
      residual[i] += values[j] * primal_vars[indices[j]];
    }
  }

  auto functor = cuopt::linear_programming::detail::violation<double>{};

  // Compute violation to lower/upper bound

  // std::transform can't take 3 inputs
  for (size_t i = 0; i < residual.size(); ++i) {
    viol[i] = functor(residual[i], constraint_lower_bounds[i], constraint_upper_bounds[i]);
  }

  // Compute the l2 primal residual
  double l2_primal_residual = std::accumulate(
    viol.cbegin(), viol.cend(), 0.0, [](double acc, double val) { return acc + val * val; });
  l2_primal_residual = std::sqrt(l2_primal_residual);

  EXPECT_NEAR(l2_primal_residual,
              solution.get_additional_termination_information().l2_primal_residual,
              epsilon);

  // Check if primal residual is indeed respecting the default tolerance
  pdlp_solver_settings_t solver_settings = pdlp_solver_settings_t<int, double>{};

  std::vector<double> combined_bounds(constraint_lower_bounds.size());

  std::transform(constraint_lower_bounds.cbegin(),
                 constraint_lower_bounds.cend(),
                 constraint_upper_bounds.cbegin(),
                 combined_bounds.begin(),
                 cuopt::linear_programming::detail::combine_finite_abs_bounds<double>{});

  double l2_norm_primal_right_hand_side = std::accumulate(
    combined_bounds.cbegin(), combined_bounds.cend(), 0.0, [](double acc, double val) {
      return acc + val * val;
    });
  l2_norm_primal_right_hand_side = std::sqrt(l2_norm_primal_right_hand_side);

  EXPECT_TRUE(l2_primal_residual <= solver_settings.tolerances.absolute_primal_tolerance +
                                      solver_settings.tolerances.relative_primal_tolerance *
                                        l2_norm_primal_right_hand_side);

  // Checking variable bounds

  // std::all_of would work but we would need C++23 zip views
  for (size_t i = 0; i < primal_vars.size(); ++i) {
    // Not always stricly true because we apply variable bound clamping on the scaled problem
    // After unscaling it, the variables might not respect exactly (this adding an epsilon)
    EXPECT_TRUE(primal_vars[i] >= variable_lower_bounds[i] - epsilon &&
                primal_vars[i] <= variable_upper_bounds[i] + epsilon);
  }
}

}  // namespace cuopt::linear_programming::test
