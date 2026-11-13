/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <cmath>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <limits>
#include <pdlp/solution_objectives.hpp>
#include <raft/core/handle.hpp>
#include <vector>

namespace cuopt::mathematical_optimization::test {

TEST(ReportedObjectives, BoxedLpAndOffDiagonalQp)
{
  raft::handle_t handle;
  const double inf = std::numeric_limits<double>::infinity();
  for (bool quadratic : {false, true}) {
    for (bool maximize : {false, true}) {
      for (double scale : {0.5, 2.0}) {
        SCOPED_TRACE(::testing::Message()
                     << "QP=" << quadratic << " max=" << maximize << " scale=" << scale);
        const double sign = maximize ? -1.0 : 1.0;
        optimization_problem_t<int, double> model(&handle);
        const double a[]{1, 2, 2}, lower[]{0, 0.25}, upper[]{2, inf};
        const double row_lower[]{2, -inf}, row_upper[]{4, 5};
        const int indices[]{0, 1, 0}, offsets[]{0, 2, 3};
        const double c[]{sign * (quadratic ? -9.0 : -6.0), sign * (quadratic ? 5.0 : 8.0)};
        model.set_csr_constraint_matrix(a, 3, indices, 3, offsets, 3);
        model.set_objective_coefficients(c, 2);
        model.set_variable_lower_bounds(lower, 2);
        model.set_variable_upper_bounds(upper, 2);
        model.set_constraint_lower_bounds(row_lower, 2);
        model.set_constraint_upper_bounds(row_upper, 2);
        model.set_objective_offset(7);
        model.set_objective_scaling_factor(scale);
        model.set_maximize(maximize);
        if (quadratic) {
          const double q[]{sign, sign * 0.5, sign * 0.5, sign};
          const int qi[]{0, 1, 0, 1}, qo[]{0, 2, 4};
          model.set_quadratic_objective_matrix(q, 4, qi, 4, qo, 3);
        }
        // x=(1,1) is feasible. For min, y=(2,-3), z=(-2,4) gives
        // A^T y + z=(-6,8). Choose c=that vector-Hx for exact dual equality.
        // The four row/bound contributions are 4,-15,-4,1 (sum -14).
        // The non-diagonal quadratic contributes (1/2)x^T Hx=3.
        const std::vector<double> x{1, 1}, y{2 * sign, -3 * sign}, z{-2 * sign, 4 * sign};
        const auto values = detail::evaluate_solution_objectives(model, x, y, z);
        EXPECT_DOUBLE_EQ(values.primal, scale * (sign * (quadratic ? -1.0 : 2.0) + 7));
        EXPECT_DOUBLE_EQ(values.dual, scale * (sign * (quadratic ? -17.0 : -14.0) + 7));
        EXPECT_DOUBLE_EQ(values.gap, scale * 16);
        EXPECT_DOUBLE_EQ(values.relative_gap,
                         values.gap / (1 + std::abs(values.primal) + std::abs(values.dual)));
        const auto missing =
          detail::evaluate_solution_objectives(model, x, y, std::vector<double>{});
        EXPECT_TRUE(std::isfinite(missing.primal));
        EXPECT_TRUE(std::isnan(missing.dual));
        EXPECT_TRUE(std::isnan(missing.gap));
        auto invalid_y = y;
        invalid_y[1]   = sign;  // Wrong sign on the row with an infinite lower bound.
        EXPECT_TRUE(std::isnan(detail::evaluate_solution_objectives(model, x, invalid_y, z).dual));
      }
    }
  }
}

TEST(ReportedObjectives, BarrierReturnsMeasuredNonzeroGap)
{
  raft::handle_t handle;
  optimization_problem_t<int, double> model(&handle);
  const double a[]{4, 1, 1, 2, 1, 4, 2, 1}, c[]{1.5, 2.25, 2.25, 0};
  const int indices[]{0, 1, 2, 3, 0, 1, 2, 3}, offsets[]{0, 4, 8};
  const double rhs[]{3.5, 3.25}, lower[]{0, 0, 0, 0}, upper[]{1, 1, 1, 1};
  model.set_csr_constraint_matrix(a, 8, indices, 8, offsets, 3);
  model.set_objective_coefficients(c, 4);
  model.set_variable_lower_bounds(lower, 4);
  model.set_variable_upper_bounds(upper, 4);
  model.set_constraint_lower_bounds(rhs, 2);
  model.set_constraint_upper_bounds(rhs, 2);
  model.set_objective_offset(7);
  pdlp_solver_settings_t<int, double> settings;
  settings.method         = method_t::Barrier;
  settings.presolver      = presolver_t::None;
  settings.crossover      = false;
  settings.dualize        = 0;
  settings.folding        = 0;
  settings.log_to_console = false;
  auto solution           = solve_lp(model, settings);
  ASSERT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  const auto x = cuopt::host_copy(solution.get_primal_solution(), handle.get_stream());
  const auto y = cuopt::host_copy(solution.get_dual_solution(), handle.get_stream());
  const auto z = cuopt::host_copy(solution.get_reduced_cost(), handle.get_stream());
  double p = 7, d = 7 + 3.5 * y[0] + 3.25 * y[1];
  for (int j = 0; j < 4; ++j) {
    p += c[j] * x[j];
    d += std::min(z[j], 0.0);  // [0,1] contributes 0 for z>=0, and z for z<0.
  }
  const auto& info = solution.get_additional_termination_information();
  EXPECT_NEAR(info.primal_objective, p, 1e-13);
  EXPECT_NEAR(info.dual_objective, d, 1e-13);
  EXPECT_GT(std::abs(p - d), 1e-12);
  EXPECT_NEAR(info.gap, std::abs(p - d), 1e-13);
  EXPECT_NEAR(info.relative_gap, std::abs(p - d) / (1 + std::abs(p) + std::abs(d)), 1e-14);
}
TEST(ReportedObjectives, QpApiUsesRestoredVectorsAndQuadraticTerm)
{
  raft::handle_t handle;
  for (bool maximize : {false, true}) {
    const double sign = maximize ? -1.0 : 1.0;
    optimization_problem_t<int, double> model(&handle);
    const double a[]{1, 1}, c[]{-4 * sign, -sign}, lower[]{0, 0}, upper[]{10, 10}, rhs[]{1};
    const int indices[]{0, 1}, offsets[]{0, 2};
    const double q[]{sign, .5 * sign, .5 * sign, sign};
    const int qi[]{0, 1, 0, 1}, qo[]{0, 2, 4};
    model.set_csr_constraint_matrix(a, 2, indices, 2, offsets, 2);
    model.set_objective_coefficients(c, 2);
    model.set_variable_lower_bounds(lower, 2);
    model.set_variable_upper_bounds(upper, 2);
    model.set_constraint_lower_bounds(rhs, 1);
    model.set_constraint_upper_bounds(rhs, 1);
    model.set_quadratic_objective_matrix(q, 4, qi, 4, qo, 3);
    model.set_objective_offset(7);
    model.set_objective_scaling_factor(2);
    model.set_maximize(maximize);
    pdlp_solver_settings_t<int, double> settings;
    settings.method         = method_t::Barrier;
    settings.presolver      = presolver_t::None;
    settings.crossover      = false;
    settings.log_to_console = false;
    auto solution           = solve_lp(model, settings);
    ASSERT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
    const auto x           = cuopt::host_copy(solution.get_primal_solution(), handle.get_stream());
    const auto y           = cuopt::host_copy(solution.get_dual_solution(), handle.get_stream());
    const auto z           = cuopt::host_copy(solution.get_reduced_cost(), handle.get_stream());
    const double quadratic = sign * (x[0] * x[0] + x[0] * x[1] + x[1] * x[1]);
    const double p         = 2 * (quadratic - sign * (4 * x[0] + x[1]) + 7);
    double d               = y[0] - quadratic + 7;
    for (auto value : z) {
      if (sign * value < 0) { d += 10 * value; }
    }
    d *= 2;
    const auto& info = solution.get_additional_termination_information();
    EXPECT_NEAR(info.primal_objective, p, 1e-11);
    EXPECT_NEAR(info.dual_objective, d, 1e-11);
    EXPECT_GT(std::abs(p - d), 1e-12);
    EXPECT_NEAR(info.gap, std::abs(p - d), 1e-11);
  }
}

}  // namespace cuopt::mathematical_optimization::test
