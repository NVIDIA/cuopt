/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>

#include <utilities/copy_helpers.hpp>
#include <utilities/inline_lp_test_utils.hpp>

#include <raft/core/handle.hpp>

#include <gmock/gmock.h>

#include <tuple>
#include <vector>

namespace cuopt::mathematical_optimization::test {

using testing::DoubleNear;
using testing::Pointwise;

using lp_solution_test_param_t = std::tuple<method_t, presolver_t>;

class lp_solution_test : public ::testing::TestWithParam<lp_solution_test_param_t> {};

TEST_P(lp_solution_test, returns_hand_checked_duals_and_reduced_costs)
{
  // The LP below has the unique certificate
  // x = (0.25, 0.5, 0, 1), y = (0.25, 0.5), z = (0, 0, 1, -1), objective = 1.5.
  //
  // The leading 2x2 block of A is invertible, and the nonzero reduced costs fix x2 and x3 at
  // opposite bounds, making both the primal and dual certificates unique.
  constexpr double solve_tolerance    = 1e-10;
  constexpr double check_tolerance    = 2e-6;
  constexpr double expected_objective = 1.5;

  const std::vector<double> expected_primal{0.25, 0.5, 0.0, 1.0};
  const std::vector<double> expected_dual{0.25, 0.5};
  const std::vector<double> expected_reduced_cost{0.0, 0.0, 1.0, -1.0};
  const auto [method, presolver] = GetParam();
  auto problem                   = cuopt::test::parse_inline_lp(R"LP(
Minimize
  obj: 1.5 x0 + 2.25 x1 + 2.25 x2
Subject To
  c0: 4 x0 + x1 + x2 + 2 x3 = 3.5
  c1: x0 + 4 x1 + 2 x2 + x3 = 3.25
Bounds
  0 <= x0 <= 1
  0 <= x1 <= 1
  0 <= x2 <= 1
  0 <= x3 <= 1
End
)LP");

  pdlp_solver_settings_t<int, double> settings;
  settings.method                               = method;
  settings.presolver                            = presolver;
  settings.crossover                            = false;
  settings.log_to_console                       = false;
  settings.tolerances.absolute_primal_tolerance = solve_tolerance;
  settings.tolerances.relative_primal_tolerance = 0.0;
  settings.tolerances.absolute_dual_tolerance   = solve_tolerance;
  settings.tolerances.relative_dual_tolerance   = 0.0;
  settings.tolerances.absolute_gap_tolerance    = solve_tolerance;
  settings.tolerances.relative_gap_tolerance    = 0.0;

  const raft::handle_t handle;
  auto solution = solve_lp(&handle, problem, settings);

  ASSERT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  const auto& info = solution.get_additional_termination_information();
  EXPECT_EQ(info.solved_by, method);
  if (method == method_t::PDLP) {
    EXPECT_GT(info.number_of_steps_taken, 0)
      << "PDLP must execute instead of returning a presolve-only solution";
  }

  const auto primal       = cuopt::host_copy(solution.get_primal_solution(), handle.get_stream());
  const auto dual         = cuopt::host_copy(solution.get_dual_solution(), handle.get_stream());
  const auto reduced_cost = cuopt::host_copy(solution.get_reduced_cost(), handle.get_stream());

  EXPECT_THAT(primal, Pointwise(DoubleNear(check_tolerance), expected_primal));
  EXPECT_THAT(dual, Pointwise(DoubleNear(check_tolerance), expected_dual));
  EXPECT_THAT(reduced_cost, Pointwise(DoubleNear(check_tolerance), expected_reduced_cost));
  EXPECT_NEAR(solution.get_objective_value(), expected_objective, check_tolerance);
}

INSTANTIATE_TEST_SUITE_P(
  LpMethodsAndPresolvers,
  lp_solution_test,
  ::testing::Combine(::testing::Values(method_t::PDLP, method_t::DualSimplex, method_t::Barrier),
                     ::testing::Values(presolver_t::None, presolver_t::PSLP, presolver_t::Papilo)));

}  // namespace cuopt::mathematical_optimization::test
