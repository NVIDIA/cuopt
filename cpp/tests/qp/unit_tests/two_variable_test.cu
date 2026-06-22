/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/inline_lp_test_utils.hpp>

#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

namespace cuopt::mathematical_optimization {

TEST(two_variable_test, simple_test)
{
  raft::handle_t handle;

  // Unconstrained optimum (4, 2) satisfies the constraint with slack.
  auto problem = cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: -8 x1 - 16 x2 + [ 2 x1 ^ 2 + 8 x2 ^ 2 ] / 2
Subject To
  c1: x1 + x2 >= 5
Bounds
  0 <= x1 <= 10
  0 <= x2 <= 10
End
)LP");

  auto settings = pdlp_solver_settings_t<int, double>();
  auto solution = solve_lp(&handle, problem, settings);

  EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(solution.get_objective_value(), -32.0, 1e-6);

  auto sol_vec = cuopt::host_copy(solution.get_primal_solution(), handle.get_stream());
  EXPECT_NEAR(sol_vec[0], 4.0, 1e-6);
  EXPECT_NEAR(sol_vec[1], 2.0, 1e-6);
}

}  // namespace cuopt::mathematical_optimization
