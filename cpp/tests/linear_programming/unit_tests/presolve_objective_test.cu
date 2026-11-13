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

TEST(PresolveObjective, AppliesInputScaleToEliminatedObjective)
{
  raft::handle_t handle;
  for (auto presolver : {presolver_t::None, presolver_t::Papilo, presolver_t::PSLP}) {
    for (bool maximize : {false, true}) {
      for (double scale : {0.5, 2.0, -2.0}) {
        SCOPED_TRACE(::testing::Message() << "presolver=" << static_cast<int>(presolver)
                                          << " maximize=" << maximize << " scale=" << scale);
        // min/max scale * (3*x + 7), x = 2, x <= 5. The exact objective is 13*scale.
        optimization_problem_t<int, double> model(&handle);
        const double a[]{1.0}, c[]{3.0}, bound[]{2.0}, rhs[]{5.0};
        const double lhs[]{-std::numeric_limits<double>::infinity()};
        const int index[]{0}, offsets[]{0, 1};
        model.set_csr_constraint_matrix(a, 1, index, 1, offsets, 2);
        model.set_objective_coefficients(c, 1);
        model.set_variable_lower_bounds(bound, 1);
        model.set_variable_upper_bounds(bound, 1);
        model.set_constraint_lower_bounds(lhs, 1);
        model.set_constraint_upper_bounds(rhs, 1);
        model.set_objective_offset(7.0);
        model.set_objective_scaling_factor(scale);
        model.set_maximize(maximize);
        pdlp_solver_settings_t<int, double> settings;
        settings.method         = method_t::PDLP;
        settings.presolver      = presolver;
        settings.log_to_console = false;
        auto result             = solve_lp(model, settings);
        ASSERT_EQ(result.get_termination_status(), pdlp_termination_status_t::Optimal);
        const auto x = cuopt::host_copy(result.get_primal_solution(), handle.get_stream());
        ASSERT_EQ(x.size(), 1);
        EXPECT_NEAR(x[0], 2.0, 1e-10);
        const auto& info = result.get_additional_termination_information();
        EXPECT_NEAR(info.primal_objective, scale * (3.0 * x[0] + 7.0), 1e-9);
        EXPECT_NEAR(info.dual_objective, 13.0 * scale, 1e-9);
        if (presolver != presolver_t::None) { EXPECT_EQ(info.number_of_steps_taken, 0); }
      }
    }
  }
}
}  // namespace cuopt::mathematical_optimization::test
