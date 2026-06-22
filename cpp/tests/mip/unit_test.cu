/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/mip_scaling_strategy.cuh>
#include <pdlp/utilities/problem_checking.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/inline_lp_test_utils.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

namespace cuopt::mathematical_optimization::test {

io::mps_data_model_t<int, double> create_std_lp_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: 1.2 x1 + 1.7 x2
Subject To
  c1_ub: x1 + x2 <= 5000
  c1_lb: x1 + x2 >= 0
Bounds
  0 <= x1 <= 3000
  0 <= x2 <= 5000
End
)LP");
}

io::mps_data_model_t<int, double> create_single_var_lp_problem()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: -0.23 x
Subject To
  c1: x = 0
Bounds
  x = 0
End
)LP");
}

io::mps_data_model_t<int, double> create_std_milp_problem(bool maximize)
{
  auto problem = create_std_lp_problem();
  problem.set_maximize(maximize);
  std::vector<char> var_types = {'I', 'C'};
  problem.set_variable_types(var_types);
  return problem;
}

io::mps_data_model_t<int, double> create_single_var_milp_problem(bool maximize)
{
  auto problem = create_single_var_lp_problem();
  problem.set_maximize(maximize);
  std::vector<char> var_types = {'I'};
  problem.set_variable_types(var_types);
  return problem;
}

TEST(LPTest, TestSampleLP2)
{
  raft::handle_t handle;

  // Two identical row constraints exercise duplicate-row handling.
  auto problem = cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: x
Subject To
  c1: x <= 1
  c2: x <= 1
End
)LP");

  cuopt::mathematical_optimization::pdlp_solver_settings_t<int, double> settings{};
  settings.set_optimality_tolerance(1e-2);
  settings.method     = cuopt::mathematical_optimization::method_t::PDLP;
  settings.time_limit = 5;

  // Solve
  auto result = cuopt::mathematical_optimization::solve_lp(&handle, problem, settings);

  // Check results
  EXPECT_EQ(result.get_termination_status(),
            cuopt::mathematical_optimization::pdlp_termination_status_t::Optimal);
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

  cuopt::mathematical_optimization::pdlp_solver_settings_t<int, double> settings{};
  settings.set_optimality_tolerance(1e-4);
  settings.time_limit = 5;
  settings.presolver  = cuopt::mathematical_optimization::presolver_t::None;

  auto result = cuopt::mathematical_optimization::solve_lp(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(),
            cuopt::mathematical_optimization::pdlp_termination_status_t::Optimal);
}

TEST(ErrorTest, TestError)
{
  raft::handle_t handle;
  auto problem = create_std_milp_problem(false);

  cuopt::mathematical_optimization::mip_solver_settings_t<int, double> settings{};
  settings.time_limit = 5;
  settings.presolver  = cuopt::mathematical_optimization::presolver_t::None;

  // Set constraint bounds
  std::vector<double> lower_bounds = {1.0};
  std::vector<double> upper_bounds = {1.0, 1.0};
  problem.set_constraint_lower_bounds(lower_bounds);
  problem.set_constraint_upper_bounds(upper_bounds);

  auto result = cuopt::mathematical_optimization::solve_mip(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(),
            cuopt::mathematical_optimization::mip_termination_status_t::NoTermination);
}

class MILPTestParams
  : public testing::TestWithParam<
      std::tuple<bool, int, bool, cuopt::mathematical_optimization::mip_termination_status_t>> {};

TEST_P(MILPTestParams, TestSampleMILP)
{
  bool maximize                    = std::get<0>(GetParam());
  int scaling                      = std::get<1>(GetParam());
  bool heuristics_only             = std::get<2>(GetParam());
  auto expected_termination_status = std::get<3>(GetParam());

  raft::handle_t handle;
  auto problem = create_std_milp_problem(maximize);

  cuopt::mathematical_optimization::mip_solver_settings_t<int, double> settings{};
  settings.time_limit      = 5;
  settings.mip_scaling     = scaling;
  settings.heuristics_only = heuristics_only;
  settings.presolver       = cuopt::mathematical_optimization::presolver_t::None;

  auto result = cuopt::mathematical_optimization::solve_mip(&handle, problem, settings);

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

  cuopt::mathematical_optimization::mip_solver_settings_t<int, double> settings{};
  settings.time_limit      = 5;
  settings.mip_scaling     = scaling;
  settings.heuristics_only = heuristics_only;
  settings.presolver       = cuopt::mathematical_optimization::presolver_t::None;

  auto result = cuopt::mathematical_optimization::solve_mip(&handle, problem, settings);

  EXPECT_EQ(result.get_termination_status(),
            cuopt::mathematical_optimization::mip_termination_status_t::Optimal);
}

INSTANTIATE_TEST_SUITE_P(
  MILPTests,
  MILPTestParams,
  testing::Values(
    std::make_tuple(true,
                    CUOPT_MIP_SCALING_ON,
                    true,
                    cuopt::mathematical_optimization::mip_termination_status_t::Optimal),
    std::make_tuple(false,
                    CUOPT_MIP_SCALING_ON,
                    false,
                    cuopt::mathematical_optimization::mip_termination_status_t::Optimal),
    std::make_tuple(true,
                    CUOPT_MIP_SCALING_OFF,
                    true,
                    cuopt::mathematical_optimization::mip_termination_status_t::Optimal),
    std::make_tuple(false,
                    CUOPT_MIP_SCALING_OFF,
                    false,
                    cuopt::mathematical_optimization::mip_termination_status_t::Optimal)));

// ---------------------------------------------------------------------------
// Scaling integrality preservation test
// ---------------------------------------------------------------------------

// Coefficient spread (~log2(100000/1) ≈ 17) exceeds the scaler's 12-threshold
// so the scaling path is exercised; row 4 omits x3 so the integer-only row
// stays integer.
static io::mps_data_model_t<int, double> create_wide_spread_milp()
{
  return cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: x0 + 2 x1 + 3 x2 + 0.5 x3
Subject To
  c0: 3 x0 + 7 x1 + 2 x2 + 1.5 x3 <= 1e6
  c1: 100000 x0 + 50000 x1 + 25000 x2 + 999.9 x3 <= 1e8
  c2: 5 x0 + 11 x1 + 13 x2 + 0.3 x3 <= 1e4
  c3: 60000 x0 + 30000 x1 + 9000 x2 + 42.42 x3 <= 1e8
  c4: x0 + x1 + x2 <= 100
  c5: 8 x0 + 4 x1 + 6 x2 + 3.14 x3 <= 1e4
Bounds
  0 <= x0 <= 1000
  0 <= x1 <= 1000
  0 <= x2 <= 1000
  0 <= x3 <= 1e6
Generals
  x0
  x1
  x2
End
)LP");
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

  mip::mip_scaling_strategy_t<int, double> scaling(op_problem);
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

  mip::mip_scaling_strategy_t<int, double> scaling(op_problem);
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

}  // namespace cuopt::mathematical_optimization::test
