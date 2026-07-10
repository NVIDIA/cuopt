/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/inline_lp_test_utils.hpp>

#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>

namespace cuopt::mathematical_optimization {

TEST(no_constraints_test, simple_test)
{
  raft::handle_t handle;

  auto problem = cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: [ 2 x1 ^ 2 + 2 x2 ^ 2 ] / 2
Bounds
  x1 free
  x2 free
End
)LP");

  auto settings = pdlp_solver_settings_t<int, double>();
  auto solution = solve_lp(&handle, problem, settings);

  EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(solution.get_objective_value(), 0.0, 1e-6);

  auto sol_vec = cuopt::host_copy(solution.get_primal_solution(), handle.get_stream());
  EXPECT_NEAR(sol_vec[0], 0.0, 1e-6);
  EXPECT_NEAR(sol_vec[1], 0.0, 1e-6);
}

}  // namespace cuopt::mathematical_optimization
