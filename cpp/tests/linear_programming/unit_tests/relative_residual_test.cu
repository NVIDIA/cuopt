/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <cmath>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <limits>
#include <raft/core/handle.hpp>
#include <utilities/copy_helpers.hpp>
#include <vector>

namespace cuopt::mathematical_optimization::test {

TEST(RelativeResiduals, PairPrimalWithRhsAndDualWithObjective)
{
  raft::handle_t handle;
  bool saw_nonzero_primal = false, saw_nonzero_dual = false;
  for (double rhs_scale : {1.0, 1000.0}) {
    optimization_problem_t<int, double> model(&handle);
    const double a[]{4, 1, 1, 2, 1, 4, 2, 1};
    const int index[]{0, 1, 2, 3, 0, 1, 2, 3}, offsets[]{0, 4, 8};
    const double c[]{0.0015, 0.00225, 0.00225, 0};
    const double rhs[]{3.5 * rhs_scale, 3.25 * rhs_scale};
    const double lower[]{0, 0, 0, 0}, upper[]{rhs_scale, rhs_scale, rhs_scale, rhs_scale};
    model.set_csr_constraint_matrix(a, 8, index, 8, offsets, 3);
    model.set_objective_coefficients(c, 4);
    model.set_variable_lower_bounds(lower, 4);
    model.set_variable_upper_bounds(upper, 4);
    model.set_constraint_lower_bounds(rhs, 2);
    model.set_constraint_upper_bounds(rhs, 2);
    pdlp_solver_settings_t<int, double> settings;
    settings.method         = method_t::Barrier;
    settings.presolver      = presolver_t::None;
    settings.crossover      = false;
    settings.dualize        = 0;
    settings.folding        = 0;
    settings.log_to_console = false;
    auto result             = solve_lp(model, settings);
    ASSERT_EQ(result.get_termination_status(), pdlp_termination_status_t::Optimal);
    const auto& info            = result.get_additional_termination_information();
    const double rhs_norm       = std::hypot(rhs[0], rhs[1]);
    const double objective_norm = std::hypot(c[0], std::hypot(c[1], c[2]));
    ASSERT_GT(rhs_norm / objective_norm, 1000.0);
    ASSERT_TRUE(std::isfinite(info.l2_primal_residual));
    ASSERT_TRUE(std::isfinite(info.l2_dual_residual));
    saw_nonzero_primal |= info.l2_primal_residual > 0;
    saw_nonzero_dual |= info.l2_dual_residual > 0;
    // Check the reporting normalization independently of the solver's residual norm.
    EXPECT_NEAR(info.l2_relative_primal_residual * (1 + rhs_norm),
                info.l2_primal_residual,
                std::abs(info.l2_primal_residual) * 1e-12);
    EXPECT_NEAR(info.l2_relative_dual_residual * (1 + objective_norm),
                info.l2_dual_residual,
                std::abs(info.l2_dual_residual) * 1e-12);
  }
  EXPECT_TRUE(saw_nonzero_primal);
  EXPECT_TRUE(saw_nonzero_dual);
}
}  // namespace cuopt::mathematical_optimization::test
