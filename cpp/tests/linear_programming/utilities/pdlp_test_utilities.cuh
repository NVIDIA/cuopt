/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>

#include <mps_parser_internal.hpp>
#include <pdlp/solve.cuh>
#include <pdlp/utils.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace cuopt::mathematical_optimization::test {
constexpr double tolerance = 1e-6f;

static std::string make_path_absolute(const std::string& file)
{
  std::string rel_file{};
  // assume relative paths are relative to RAPIDS_DATASET_ROOT_DIR
  const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
  rel_file                                = rapidsDatasetRootDir + "/" + file;
  return rel_file;
}

// Wrapper for the batch PDLP flow: convert and potentially expand the problem and call
// run_batch_pdlp.
template <typename i_t, typename f_t>
static cuopt::mathematical_optimization::optimization_problem_solution_t<i_t, f_t> solve_lp_batch(
  raft::handle_t const* handle_ptr,
  const cuopt::mathematical_optimization::io::mps_data_model_t<i_t, f_t>& mps_data_model,
  const cuopt::mathematical_optimization::pdlp_solver_settings_t<i_t, f_t>& settings)
{
  auto gpu_op = cuopt::mathematical_optimization::mps_data_model_to_optimization_problem<i_t, f_t>(
    handle_ptr, mps_data_model);
  auto batch_settings                                = settings;
  batch_settings.generate_batch_primal_dual_solution = true;
  return cuopt::mathematical_optimization::run_batch_pdlp(gpu_op, batch_settings);
}

// Overwrites the device_uvector with the host-side contents, resizing as needed.
template <typename f_t>
static void assign_device_uvector_from_host(rmm::device_uvector<f_t>& target,
                                            const std::vector<f_t>& src,
                                            rmm::cuda_stream_view stream)
{
  target.resize(src.size(), stream);
  raft::copy(target.data(), src.data(), src.size(), stream);
}

// Convenience wrapper for the fixed-path batch PDLP flow:
// parse → convert MPS to optimization_problem_t → pre-expand any per-climber problem fields
// (objective coefficients, constraint lower/upper bounds, objective offsets) on the
// optimization_problem_t → dispatch to `run_batch_pdlp` with fixed_batch_size set (fixed path).
//
// Any of the per_climber_* vectors may be empty to skip that expansion. The vectors use the
// same flat COL-major layout the solver expects internally:
//   - per_climber_objective_coefficients: size (batch_size * n_variables), block per climber.
//   - per_climber_constraint_lower_bounds / upper_bounds: size (batch_size * n_constraints).
//   - per_climber_objective_offsets: size (batch_size).
template <typename i_t, typename f_t>
static cuopt::mathematical_optimization::optimization_problem_solution_t<i_t, f_t>
solve_lp_batch_fixed(
  raft::handle_t const* handle_ptr,
  const cuopt::mathematical_optimization::io::mps_data_model_t<i_t, f_t>& mps_data_model,
  cuopt::mathematical_optimization::pdlp_solver_settings_t<i_t, f_t> settings,
  i_t batch_size,
  const std::vector<f_t>& per_climber_objective_coefficients  = {},
  const std::vector<f_t>& per_climber_constraint_lower_bounds = {},
  const std::vector<f_t>& per_climber_constraint_upper_bounds = {},
  const std::vector<f_t>& per_climber_objective_offsets       = {},
  bool use_direct_api                                         = false)
{
  auto gpu_op = cuopt::mathematical_optimization::mps_data_model_to_optimization_problem<i_t, f_t>(
    handle_ptr, mps_data_model);
  auto stream = handle_ptr->get_stream();

  if (!per_climber_objective_coefficients.empty()) {
    assign_device_uvector_from_host(
      gpu_op.get_objective_coefficients(), per_climber_objective_coefficients, stream);
  }

  if (!per_climber_constraint_lower_bounds.empty()) {
    assign_device_uvector_from_host(
      gpu_op.get_constraint_lower_bounds(), per_climber_constraint_lower_bounds, stream);
  }

  if (!per_climber_constraint_upper_bounds.empty()) {
    assign_device_uvector_from_host(
      gpu_op.get_constraint_upper_bounds(), per_climber_constraint_upper_bounds, stream);
  }

  if (!per_climber_objective_offsets.empty()) {
    gpu_op.set_batch_objective_offsets(per_climber_objective_offsets);
  }

  settings.generate_batch_primal_dual_solution = true;
  settings.fixed_batch_size                    = batch_size;
  if (use_direct_api) {
    return cuopt::mathematical_optimization::solve_lp(gpu_op, settings, false);
  }
  return cuopt::mathematical_optimization::run_batch_pdlp(gpu_op, settings);
}

// Compute on the CPU x * c to check that the returned objective value is correct
static void test_objective_sanity(
  const cuopt::mathematical_optimization::io::mps_data_model_t<int, double>& op_problem,
  const rmm::device_uvector<double>& primal_solution,
  double objective_value,
  double epsilon = tolerance)
{
  const auto primal_vars = host_copy(primal_solution, primal_solution.stream());
  const auto& c_vector   = op_problem.get_objective_coefficients();
  if (primal_vars.size() != c_vector.size()) {
    EXPECT_EQ(primal_vars.size(), c_vector.size());
    return;
  }
  std::vector<double> out(primal_vars.size());
  std::transform(primal_vars.cbegin(),
                 primal_vars.cend(),
                 c_vector.cbegin(),
                 out.begin(),
                 std::multiplies<double>());

  double sum = std::reduce(out.cbegin(), out.cend(), 0.0);

  EXPECT_NEAR(sum, objective_value, epsilon);
}

// Compute on the CPU x * c to check that the returned objective value is correct
static void test_objective_sanity(
  const cuopt::mathematical_optimization::io::mps_data_model_t<int, double>& op_problem,
  const std::vector<double>& primal_solution,
  double objective_value,
  double epsilon = tolerance)
{
  const auto& c_vector = op_problem.get_objective_coefficients();
  if (primal_solution.size() != c_vector.size()) {
    EXPECT_EQ(primal_solution.size(), c_vector.size());
    return;
  }
  std::vector<double> out(primal_solution.size());
  std::transform(primal_solution.cbegin(),
                 primal_solution.cend(),
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
  const cuopt::mathematical_optimization::io::mps_data_model_t<int, double>& op_problem,
  const optimization_problem_solution_t<int, double>::additional_termination_information_t&
    termination_information,
  const rmm::device_uvector<double>& primal_solution,
  double epsilon        = tolerance,
  bool presolve_enabled = false)
{
  const std::vector<double> primal_vars = host_copy(primal_solution, primal_solution.stream());
  const std::vector<double>& values     = op_problem.get_constraint_matrix_values();
  const std::vector<int>& indices       = op_problem.get_constraint_matrix_indices();
  const std::vector<int>& offsets       = op_problem.get_constraint_matrix_offsets();
  const std::vector<double>& constraint_lower_bounds = op_problem.get_constraint_lower_bounds();
  const std::vector<double>& constraint_upper_bounds = op_problem.get_constraint_upper_bounds();
  const std::vector<double>& variable_lower_bounds   = op_problem.get_variable_lower_bounds();
  const std::vector<double>& variable_upper_bounds   = op_problem.get_variable_upper_bounds();
  std::vector<double> residual(constraint_lower_bounds.size(), 0.0);
  std::vector<double> viol(constraint_lower_bounds.size(), 0.0);

  // No dual solution and residual for presolved problems
  if (!presolve_enabled) {
    // CSR SpMV
    for (size_t i = 0; i < offsets.size() - 1; ++i) {
      for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
        residual[i] += values[j] * primal_vars[indices[j]];
      }
    }

    auto functor = cuopt::mathematical_optimization::pdlp::violation<double>{};

    // Compute violation to lower/upper bound

    // std::transform can't take 3 inputs
    for (size_t i = 0; i < residual.size(); ++i) {
      viol[i] = functor(residual[i], constraint_lower_bounds[i], constraint_upper_bounds[i]);
    }

    // Compute the l2 primal residual
    double l2_primal_residual = std::accumulate(
      viol.cbegin(), viol.cend(), 0.0, [](double acc, double val) { return acc + val * val; });
    l2_primal_residual = std::sqrt(l2_primal_residual);

    EXPECT_NEAR(l2_primal_residual, termination_information.l2_primal_residual, epsilon);

    // Check if primal residual is indeed respecting the default tolerance
    pdlp_solver_settings_t solver_settings = pdlp_solver_settings_t<int, double>{};
    solver_settings.set_optimality_tolerance(epsilon);

    std::vector<double> combined_bounds(constraint_lower_bounds.size());

    std::transform(constraint_lower_bounds.cbegin(),
                   constraint_lower_bounds.cend(),
                   constraint_upper_bounds.cbegin(),
                   combined_bounds.begin(),
                   cuopt::mathematical_optimization::pdlp::combine_finite_abs_bounds<double>{});

    double l2_norm_primal_right_hand_side = std::accumulate(
      combined_bounds.cbegin(), combined_bounds.cend(), 0.0, [](double acc, double val) {
        return acc + val * val;
      });
    l2_norm_primal_right_hand_side = std::sqrt(l2_norm_primal_right_hand_side);

    EXPECT_TRUE(l2_primal_residual <= solver_settings.tolerances.absolute_primal_tolerance +
                                        solver_settings.tolerances.relative_primal_tolerance *
                                          l2_norm_primal_right_hand_side);
  }

  // Checking variable bounds

  // std::all_of would work but we would need C++23 zip views
  for (size_t i = 0; i < primal_vars.size(); ++i) {
    // Not always strictly true because we apply variable bound clamping on the scaled problem
    // After unscaling it, the variables might not respect exactly (this adding an epsilon)
    auto condition = primal_vars[i] >= variable_lower_bounds[i] - epsilon &&
                     primal_vars[i] <= variable_upper_bounds[i] + epsilon;
    if (!condition) {
      std::cout << "Variable " << i << " is " << primal_vars[i] << " but should be between "
                << variable_lower_bounds[i] - epsilon << " and "
                << variable_upper_bounds[i] + epsilon << std::endl;
    }
    EXPECT_TRUE(condition);
  }
}

}  // namespace cuopt::mathematical_optimization::test
