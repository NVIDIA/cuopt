/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/solve.hpp>
#include <mip_heuristics/mip_scaling_strategy.cuh>
#include <mps_parser/parser.hpp>
#include <pdlp/utilities/problem_checking.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
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

mps_parser::mps_data_model_t<int, double> create_single_var_lp_problem()
{
  mps_parser::mps_data_model_t<int, double> problem;

  // Set up constraint matrix in CSR format
  std::vector<int> offsets         = {0, 1};
  std::vector<int> indices         = {0};
  std::vector<double> coefficients = {1.0};
  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());

  // Set constraint bounds
  std::vector<double> lower_bounds = {0.0};
  std::vector<double> upper_bounds = {0.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());

  // Set variable bounds
  std::vector<double> var_lower = {0.0};
  std::vector<double> var_upper = {0.0};
  problem.set_variable_lower_bounds(var_lower.data(), var_lower.size());
  problem.set_variable_upper_bounds(var_upper.data(), var_upper.size());

  // Set objective coefficients
  std::vector<double> obj_coeffs = {-0.23};
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

// Create standard MILP test problem matching Python test
mps_parser::mps_data_model_t<int, double> create_single_var_milp_problem(bool maximize)
{
  auto problem = create_single_var_lp_problem();

  // Set variable types for MILP
  std::vector<char> var_types = {'I'};
  problem.set_variable_types(var_types);
  problem.set_maximize(maximize);

  return problem;
}

TEST(LPTest, TestSampleLP2)
{
  raft::handle_t handle;

  // Construct a simple LP problem:
  // Minimize:    x
  // Subject to:  x <= 1
  //              x <= 1
  //              x >= 0

  // One variable, two constraints (both x <= 1)
  std::vector<double> A_values = {1.0, 1.0};
  std::vector<int> A_indices   = {0, 0};
  std::vector<int> A_offsets   = {0, 1, 2};  // CSR: 2 constraints, 1 variable

  std::vector<double> b       = {1.0, 1.0};  // RHS for both constraints
  std::vector<double> b_lower = {-std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity()};

  std::vector<double> c = {1.0};  // Objective: Minimize x

  std::vector<char> row_types = {'L', 'L'};  // Both constraints are <=

  // Build the problem
  mps_parser::mps_data_model_t<int, double> problem;
  problem.set_csr_constraint_matrix(A_values.data(),
                                    A_values.size(),
                                    A_indices.data(),
                                    A_indices.size(),
                                    A_offsets.data(),
                                    A_offsets.size());
  problem.set_constraint_upper_bounds(b.data(), b.size());
  problem.set_constraint_lower_bounds(b_lower.data(), b_lower.size());

  // Set variable bounds (x >= 0)
  std::vector<double> var_lower = {0.0};
  std::vector<double> var_upper = {std::numeric_limits<double>::infinity()};
  problem.set_variable_lower_bounds(var_lower.data(), var_lower.size());
  problem.set_variable_upper_bounds(var_upper.data(), var_upper.size());

  problem.set_objective_coefficients(c.data(), c.size());
  problem.set_maximize(false);
  // Set up solver settings
  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings{};
  settings.set_optimality_tolerance(1e-2);
  settings.method     = cuopt::linear_programming::method_t::PDLP;
  settings.time_limit = 5;

  // Solve
  auto result = cuopt::linear_programming::solve_lp(&handle, problem, settings);

  // Check results
  EXPECT_EQ(result.get_termination_status(),
            cuopt::linear_programming::pdlp_termination_status_t::Optimal);
  ASSERT_EQ(result.get_primal_solution().size(), 1);

  // Copy solution to host to access values
  auto primal_host = cuopt::host_copy(result.get_primal_solution(), handle.get_stream());
  EXPECT_NEAR(primal_host[0], 0.0, 1e-6);

  EXPECT_NEAR(result.get_additional_termination_information().primal_objective, 0.0, 1e-6);
  EXPECT_NEAR(result.get_additional_termination_information().dual_objective, 0.0, 1e-6);
}

TEST(LPTest, TestSampleLP)
{
  raft::handle_t handle;
  auto problem = create_std_lp_problem();

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings{};
  settings.set_optimality_tolerance(1e-4);
  settings.time_limit = 5;
  settings.presolver  = cuopt::linear_programming::presolver_t::None;

  auto result = cuopt::linear_programming::solve_lp(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(),
            cuopt::linear_programming::pdlp_termination_status_t::Optimal);
}

TEST(ErrorTest, TestError)
{
  raft::handle_t handle;
  auto problem = create_std_milp_problem(false);

  cuopt::linear_programming::mip_solver_settings_t<int, double> settings{};
  settings.time_limit = 5;
  settings.presolver  = cuopt::linear_programming::presolver_t::None;

  // Set constraint bounds
  std::vector<double> lower_bounds = {1.0};
  std::vector<double> upper_bounds = {1.0, 1.0};
  problem.set_constraint_lower_bounds(lower_bounds.data(), lower_bounds.size());
  problem.set_constraint_upper_bounds(upper_bounds.data(), upper_bounds.size());

  auto result = cuopt::linear_programming::solve_mip(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(),
            cuopt::linear_programming::mip_termination_status_t::NoTermination);
}

class MILPTestParams
  : public testing::TestWithParam<
      std::tuple<bool, int, bool, cuopt::linear_programming::mip_termination_status_t>> {};

TEST_P(MILPTestParams, TestSampleMILP)
{
  bool maximize                    = std::get<0>(GetParam());
  int scaling                      = std::get<1>(GetParam());
  bool heuristics_only             = std::get<2>(GetParam());
  auto expected_termination_status = std::get<3>(GetParam());

  raft::handle_t handle;
  auto problem = create_std_milp_problem(maximize);

  cuopt::linear_programming::mip_solver_settings_t<int, double> settings{};
  settings.time_limit      = 5;
  settings.mip_scaling     = scaling;
  settings.heuristics_only = heuristics_only;
  settings.presolver       = cuopt::linear_programming::presolver_t::None;

  auto result = cuopt::linear_programming::solve_mip(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(), expected_termination_status);
}

TEST_P(MILPTestParams, TestSingleVarMILP)
{
  bool maximize                    = std::get<0>(GetParam());
  int scaling                      = std::get<1>(GetParam());
  bool heuristics_only             = std::get<2>(GetParam());
  auto expected_termination_status = std::get<3>(GetParam());

  raft::handle_t handle;
  auto problem = create_single_var_milp_problem(maximize);

  cuopt::linear_programming::mip_solver_settings_t<int, double> settings{};
  settings.time_limit      = 5;
  settings.mip_scaling     = scaling;
  settings.heuristics_only = heuristics_only;
  settings.presolver       = cuopt::linear_programming::presolver_t::None;

  auto result = cuopt::linear_programming::solve_mip(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(),
            cuopt::linear_programming::mip_termination_status_t::Optimal);
}

INSTANTIATE_TEST_SUITE_P(
  MILPTests,
  MILPTestParams,
  testing::Values(std::make_tuple(true,
                                  CUOPT_MIP_SCALING_ON,
                                  true,
                                  cuopt::linear_programming::mip_termination_status_t::Optimal),
                  std::make_tuple(false,
                                  CUOPT_MIP_SCALING_ON,
                                  false,
                                  cuopt::linear_programming::mip_termination_status_t::Optimal),
                  std::make_tuple(true,
                                  CUOPT_MIP_SCALING_OFF,
                                  true,
                                  cuopt::linear_programming::mip_termination_status_t::Optimal),
                  std::make_tuple(false,
                                  CUOPT_MIP_SCALING_OFF,
                                  false,
                                  cuopt::linear_programming::mip_termination_status_t::Optimal)));

// ---------------------------------------------------------------------------
// Scaling integrality preservation test
// ---------------------------------------------------------------------------

static mps_parser::mps_data_model_t<int, double> create_wide_spread_milp()
{
  mps_parser::mps_data_model_t<int, double> problem;

  // 6 rows, 4 variables (x0=INT, x1=INT, x2=INT, x3=CONT)
  // Coefficient spread: ~log2(100000/1) ≈ 17, well above the 12-threshold.
  // clang-format off
  std::vector<double> values = {
    3.0, 7.0, 2.0, 1.5,          // row 0: small ints + cont
    100000.0, 50000.0, 25000.0, 999.9, // row 1: large ints + cont
    5.0, 11.0, 13.0, 0.3,        // row 2: small primes + cont
    60000.0, 30000.0, 9000.0, 42.42,   // row 3: large + cont
    1.0, 1.0, 1.0, 0.0,          // row 4: unit row (no cont)
    8.0, 4.0, 6.0, 3.14          // row 5: small ints + cont
  };
  // clang-format on
  std::vector<int> indices = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                              0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  std::vector<int> offsets = {0, 4, 8, 12, 16, 20, 24};
  problem.set_csr_constraint_matrix(
    values.data(), values.size(), indices.data(), indices.size(), offsets.data(), offsets.size());

  std::vector<double> cl = {0, 0, 0, 0, 0, 0};
  std::vector<double> cu = {1e6, 1e8, 1e4, 1e8, 100, 1e4};
  problem.set_constraint_lower_bounds(cl.data(), cl.size());
  problem.set_constraint_upper_bounds(cu.data(), cu.size());

  std::vector<double> vl = {0, 0, 0, 0};
  std::vector<double> vu = {1000, 1000, 1000, 1e6};
  problem.set_variable_lower_bounds(vl.data(), vl.size());
  problem.set_variable_upper_bounds(vu.data(), vu.size());

  std::vector<double> obj = {1.0, 2.0, 3.0, 0.5};
  problem.set_objective_coefficients(obj.data(), obj.size());
  problem.set_maximize(false);

  std::vector<char> var_types = {'I', 'I', 'I', 'C'};
  problem.set_variable_types(var_types);

  return problem;
}

TEST(ScalingIntegrity, IntegerCoefficientsPreservedAfterScaling)
{
  raft::handle_t handle;
  auto mps_problem = create_wide_spread_milp();
  auto op_problem  = mps_data_model_to_optimization_problem(&handle, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  const int nnz = op_problem.get_nnz();

  auto pre_values =
    cuopt::host_copy(op_problem.get_constraint_matrix_values(), handle.get_stream());
  auto col_indices =
    cuopt::host_copy(op_problem.get_constraint_matrix_indices(), handle.get_stream());
  auto var_types = cuopt::host_copy(op_problem.get_variable_types(), handle.get_stream());
  handle.sync_stream();

  std::vector<bool> was_integer(nnz, false);
  for (int k = 0; k < nnz; ++k) {
    int col = col_indices[k];
    if (var_types[col] == var_t::INTEGER) {
      double abs_val = std::abs(pre_values[k]);
      if (abs_val > 0.0 &&
          std::abs(abs_val - std::round(abs_val)) <= 1e-6 * std::max(1.0, abs_val)) {
        was_integer[k] = true;
      }
    }
  }

  detail::mip_scaling_strategy_t<int, double> scaling(op_problem);
  scaling.scale_problem();

  auto post_values =
    cuopt::host_copy(op_problem.get_constraint_matrix_values(), handle.get_stream());
  handle.sync_stream();

  int violations = 0;
  for (int k = 0; k < nnz; ++k) {
    if (!was_integer[k]) { continue; }
    double abs_val  = std::abs(post_values[k]);
    double frac_err = std::abs(abs_val - std::round(abs_val));
    double rel_tol  = 1e-6 * std::max(1.0, abs_val);
    if (frac_err > rel_tol) {
      ++violations;
      ADD_FAILURE() << "Coefficient [" << k << "] col=" << col_indices[k] << " was integer ("
                    << pre_values[k] << ") but after scaling is " << post_values[k]
                    << " (frac_err=" << frac_err << ")";
    }
  }
  EXPECT_EQ(violations, 0) << violations << " integer coefficients lost integrality after scaling";
}

TEST(ScalingIntegrity, NoObjectiveScalingPreservesIntegerCoefficients)
{
  raft::handle_t handle;
  auto mps_problem = create_wide_spread_milp();
  auto op_problem  = mps_data_model_to_optimization_problem(&handle, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  const int nnz = op_problem.get_nnz();

  auto pre_values =
    cuopt::host_copy(op_problem.get_constraint_matrix_values(), handle.get_stream());
  auto col_indices =
    cuopt::host_copy(op_problem.get_constraint_matrix_indices(), handle.get_stream());
  auto var_types = cuopt::host_copy(op_problem.get_variable_types(), handle.get_stream());
  handle.sync_stream();

  std::vector<bool> was_integer(nnz, false);
  for (int k = 0; k < nnz; ++k) {
    int col = col_indices[k];
    if (var_types[col] == var_t::INTEGER) {
      double abs_val = std::abs(pre_values[k]);
      if (abs_val > 0.0 &&
          std::abs(abs_val - std::round(abs_val)) <= 1e-6 * std::max(1.0, abs_val)) {
        was_integer[k] = true;
      }
    }
  }

  detail::mip_scaling_strategy_t<int, double> scaling(op_problem);
  scaling.scale_problem(/*scale_objective=*/false);

  auto post_values =
    cuopt::host_copy(op_problem.get_constraint_matrix_values(), handle.get_stream());
  handle.sync_stream();

  int violations = 0;
  for (int k = 0; k < nnz; ++k) {
    if (!was_integer[k]) { continue; }
    double abs_val  = std::abs(post_values[k]);
    double frac_err = std::abs(abs_val - std::round(abs_val));
    double rel_tol  = 1e-6 * std::max(1.0, abs_val);
    if (frac_err > rel_tol) { ++violations; }
  }
  EXPECT_EQ(violations, 0) << violations
                           << " integer coefficients lost integrality after scaling (no-obj mode)";
}

}  // namespace cuopt::linear_programming::test
