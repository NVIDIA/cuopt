/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <cuopt/mathematical_optimization/solver_settings.hpp>
#include <utilities/copy_helpers.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <stdexcept>
#include <vector>

namespace cuopt::mathematical_optimization::test {
namespace {

using pdlp::reduced_matrix_mode_t;

pdlp_solver_settings_t<int, double> make_settings(reduced_matrix_mode_t mode)
{
  pdlp_solver_settings_t<int, double> settings;
  settings.method                      = method_t::PDLP;
  settings.presolver                   = presolver_t::None;
  settings.crossover                   = false;
  settings.log_to_console              = false;
  settings.iteration_limit             = 20000;
  settings.time_limit                  = 30;
  settings.hyper_params.reduced_matrix = mode;
  settings.set_optimality_tolerance(1e-8);
  return settings;
}

io::mps_data_model_t<int, double> make_reduction_test_model()
{
  io::mps_data_model_t<int, double> model;
  model.set_maximize(false);
  const std::vector<double> values{1, 2, 3, 4};
  const std::vector<int> columns{0, 1, 1, 2}, offsets{0, 2, 4};
  const std::vector<double> objective{1, 2, 3}, lower{0, 0, 0}, upper{10, 10, 10};
  const std::vector<double> row_bounds{2, 3};
  model.set_csr_constraint_matrix(values, columns, offsets);
  model.set_objective_coefficients(objective);
  model.set_variable_lower_bounds(lower);
  model.set_variable_upper_bounds(upper);
  model.set_constraint_lower_bounds(row_bounds);
  model.set_constraint_upper_bounds(row_bounds);
  return model;
}

io::mps_data_model_t<int, double> make_bound_dominated_model()
{
  // The optimum has x[0] = 3, x[1] = 2, and the other fourteen columns at alternating
  // nonzero lower/upper bounds. Dual y = (1, 1) certifies this optimum: reduced costs
  // are zero for columns 0 and 1 and +10/-10 for the remaining columns.
  constexpr int n = 16;
  std::vector<double> values(2 * n, 1.0), objective(n), lower(n, -2.0), upper(n, 3.0);
  std::vector<int> columns(2 * n), offsets{0, n, 2 * n};
  for (int j = 0; j < n; ++j) {
    columns[j] = columns[n + j] = j;
    values[n + j]               = j == 1 ? 1.1 : (j % 2 == 0 ? 1.0 : 2.0);
    objective[j]                = 1.0 + values[n + j];
    if (j >= 2) { objective[j] += j % 2 == 0 ? 10.0 : -10.0; }
  }
  lower[0] = lower[1] = 0.0;
  upper[0] = upper[1] = 10.0;
  const std::vector<double> row_bounds{12.0, 33.2};
  io::mps_data_model_t<int, double> model;
  model.set_maximize(false);
  model.set_csr_constraint_matrix(values, columns, offsets);
  model.set_objective_coefficients(objective);
  model.set_variable_lower_bounds(lower);
  model.set_variable_upper_bounds(upper);
  model.set_constraint_lower_bounds(row_bounds);
  model.set_constraint_upper_bounds(row_bounds);
  return model;
}

void set_near_optimal_start(pdlp_solver_settings_t<int, double>& settings,
                            const io::mps_data_model_t<int, double>& model,
                            const raft::handle_t& handle)
{
  auto primal = model.get_variable_lower_bounds();
  for (size_t j = 3; j < primal.size(); j += 2) {
    primal[j] = model.get_variable_upper_bounds()[j];
  }
  // Retain a small constraint residual so the solve must iterate, with most columns
  // already at their optimal bounds. The nearly parallel free columns slow convergence.
  primal[0] = 1.1 - primal[2];
  primal[1] = 1.9;
  const std::vector<double> dual{1.0, 1.0};
  settings.set_initial_primal_solution(primal.data(), primal.size(), handle.get_stream());
  settings.set_initial_dual_solution(dual.data(), dual.size(), handle.get_stream());
  handle.sync_stream();
}

void expect_optimal_solution(const io::mps_data_model_t<int, double>& model,
                             const optimization_problem_solution_t<int, double>& result,
                             double expected_objective,
                             const raft::handle_t& handle,
                             double tolerance = 1e-5)
{
  ASSERT_EQ(result.get_termination_status(), pdlp_termination_status_t::Optimal);
  const auto primal = host_copy(result.get_primal_solution(), handle.get_stream());
  const auto& costs = model.get_objective_coefficients();
  ASSERT_EQ(primal.size(), costs.size());
  EXPECT_NEAR(result.get_objective_value(), expected_objective, tolerance);
  EXPECT_NEAR(std::inner_product(costs.begin(), costs.end(), primal.begin(), 0.0),
              expected_objective,
              tolerance);
  for (size_t j = 0; j < primal.size(); ++j) {
    SCOPED_TRACE(j);
    EXPECT_GE(primal[j], model.get_variable_lower_bounds()[j] - tolerance);
    EXPECT_LE(primal[j], model.get_variable_upper_bounds()[j] + tolerance);
  }
  const auto& values  = model.get_constraint_matrix_values();
  const auto& columns = model.get_constraint_matrix_indices();
  const auto& offsets = model.get_constraint_matrix_offsets();
  for (size_t row = 0; row + 1 < offsets.size(); ++row) {
    SCOPED_TRACE(row);
    double activity = 0.0;
    for (int k = offsets[row]; k < offsets[row + 1]; ++k) {
      activity += values[k] * primal[columns[k]];
    }
    EXPECT_GE(activity, model.get_constraint_lower_bounds()[row] - tolerance);
    EXPECT_LE(activity, model.get_constraint_upper_bounds()[row] + tolerance);
  }
}

TEST(ReducedMatrixSettings, RejectsInvalidParameterValues)
{
  solver_settings_t<int, double> settings;
  for (int value : {-2, 2}) {
    SCOPED_TRACE(value);
    EXPECT_THROW(settings.set_parameter(CUOPT_PDLP_HYPER_REDUCED_MATRIX, value),
                 std::invalid_argument);
  }
  for (const char* value : {"-2", "2", "OFF", "ON", "COLUMN_REDUCTION"}) {
    SCOPED_TRACE(value);
    EXPECT_THROW(settings.set_parameter_from_string(CUOPT_PDLP_HYPER_REDUCED_MATRIX, value),
                 std::invalid_argument);
  }
}

TEST(ReducedMatrixSettings, RejectsInvalidTypedMode)
{
  raft::handle_t handle;
  auto model = make_reduction_test_model();
  for (int value : {-2, 2}) {
    SCOPED_TRACE(value);
    auto settings = make_settings(static_cast<reduced_matrix_mode_t>(value));
    auto result   = solve_lp(&handle, model, settings);
    EXPECT_EQ(result.get_error_status().get_error_type(), cuopt::error_type_t::ValidationError);
  }
}

class ReducedMatrixSolve : public testing::TestWithParam<int> {};

TEST_P(ReducedMatrixSolve, EqualityConstrainedProblem)
{
  raft::handle_t handle;
  auto model    = make_reduction_test_model();
  auto settings = make_settings(static_cast<reduced_matrix_mode_t>(GetParam()));
  auto result   = solve_lp(&handle, model, settings);
  expect_optimal_solution(model, result, 2.0, handle);
}

TEST_P(ReducedMatrixSolve, NonzeroBoundsWithFewInteriorColumns)
{
  raft::handle_t handle;
  auto model    = make_bound_dominated_model();
  auto settings = make_settings(static_cast<reduced_matrix_mode_t>(GetParam()));
  set_near_optimal_start(settings, model, handle);
  auto result = solve_lp(&handle, model, settings);
  expect_optimal_solution(model, result, -304.8, handle);
}

TEST_P(ReducedMatrixSolve, SingleClimberBoundOverrideChangesOptimum)
{
  raft::handle_t handle;
  auto model    = make_bound_dominated_model();
  auto settings = make_settings(static_cast<reduced_matrix_mode_t>(GetParam()));
  // Tightening column 2 from -2 to -1 raises the objective by 10 and changes x[0] to 2.
  settings.new_bounds.emplace_back(0, 2, -1.0, 3.0);
  settings.generate_batch_primal_dual_solution  = true;
  settings.hyper_params.do_curtis_reid_scaling  = false;
  auto expected_model                           = model;
  expected_model.get_variable_lower_bounds()[2] = -1.0;
  set_near_optimal_start(settings, expected_model, handle);
  auto result = solve_lp(&handle, model, settings);
  expect_optimal_solution(expected_model, result, -294.8, handle);
}

INSTANTIATE_TEST_SUITE_P(DisabledAndEnabled, ReducedMatrixSolve, testing::Values(0, 1));

TEST(ReducedMatrixFallback, UnsupportedRestartScalingMatchesFullSolver)
{
  raft::handle_t handle;
  auto model                                = make_reduction_test_model();
  auto settings                             = make_settings(reduced_matrix_mode_t::OFF);
  settings.iteration_limit                  = 1000;
  settings.hyper_params.rescale_for_restart = false;
  auto full                                 = solve_lp(&handle, model, settings, true, false);
  // Unsupported configurations must retain the full solver's behavior, including
  // returning the current iterate when its iteration limit is reached.
  ASSERT_TRUE(full.get_termination_status() == pdlp_termination_status_t::Optimal ||
              full.get_termination_status() == pdlp_termination_status_t::IterationLimit);
  settings.hyper_params.reduced_matrix = reduced_matrix_mode_t::COLUMN_REDUCTION;
  auto result                          = solve_lp(&handle, model, settings, true, false);
  EXPECT_EQ(result.get_termination_status(), full.get_termination_status());
  EXPECT_NEAR(result.get_objective_value(), full.get_objective_value(), 1e-9);
  const auto full_primal = host_copy(full.get_primal_solution(), handle.get_stream());
  const auto primal      = host_copy(result.get_primal_solution(), handle.get_stream());
  ASSERT_EQ(full_primal.size(), model.get_objective_coefficients().size());
  ASSERT_EQ(primal.size(), full_primal.size());
  for (size_t j = 0; j < primal.size(); ++j) {
    SCOPED_TRACE(j);
    EXPECT_NEAR(primal[j], full_primal[j], 1e-9);
  }
}

TEST(ReducedMatrixPrecision, SinglePrecisionWithAutomaticSizeSetting)
{
  raft::handle_t handle;
  auto model = make_reduction_test_model();
  for (int threshold : {-1, 0}) {
    SCOPED_TRACE(threshold);
    auto settings                = make_settings(reduced_matrix_mode_t::DEFAULT);
    settings.pdlp_precision      = pdlp_precision_t::SinglePrecision;
    settings.large_nnz_threshold = threshold;
    settings.set_optimality_tolerance(1e-5);
    auto result = solve_lp(&handle, model, settings);
    expect_optimal_solution(model, result, 2.0, handle, 1e-4);
  }
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::test
