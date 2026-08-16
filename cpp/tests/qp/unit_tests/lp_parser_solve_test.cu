/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization {

// End-to-end smoke tests that parse an LP file and solve via PDLP.
// Validates objective value and primal solution against hand-computed
// optima. The point is to verify the LP parser's quadratic-objective
// representation (upper-triangular CSR) round-trips correctly through
// cuOpt's solver (which applies H = Q + Q^T internally before solving
// (1/2) x^T H x).

namespace {

void expect_optimal_solution(const std::string& lp_text,
                             double expected_objective,
                             const std::vector<double>& expected_x)
{
  raft::handle_t handle;
  auto problem  = io::read_lp_from_string<int, double>(lp_text);
  auto settings = pdlp_solver_settings_t<int, double>();
  auto solution = solve_lp(&handle, problem, settings);

  ASSERT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(solution.get_objective_value(), expected_objective, 1e-4);

  auto sol = cuopt::host_copy(solution.get_primal_solution(), handle.get_stream());
  ASSERT_EQ(sol.size(), expected_x.size());
  for (size_t i = 0; i < expected_x.size(); ++i) {
    EXPECT_NEAR(sol[i], expected_x[i], 1e-4) << "x[" << i << "]";
  }
}

}  // namespace

// Diagonal-only quadratic objective.
// Minimize x1^2 + 4 x2^2 - 8 x1 - 16 x2 s.t. x1 + x2 >= 5, 0 <= x1, x2 <= 10.
// Unconstrained optimum (4, 2) satisfies the constraint with slack; obj = -32.
TEST(lp_parser_solve, qp_diagonal_only)
{
  expect_optimal_solution(R"LP(
Minimize
  obj: -8 x1 - 16 x2 + [ 2 x1 ^ 2 + 8 x2 ^ 2 ] / 2
Subject To
  c1: x1 + x2 >= 5
Bounds
  0 <= x1 <= 10
  0 <= x2 <= 10
End
)LP",
                          -32.0,
                          {4.0, 2.0});
}

// Quadratic objective with a cross term — exercises the upper-triangular
// off-diagonal storage path that this PR introduced.
//
// Minimize x1^2 + 2 x1 x2 + 2 x2^2 - 6 x1 - 8 x2 s.t. x1 + x2 <= 10.
// Hessian H = [[2, 2], [2, 4]] is positive definite.
// Unconstrained optimum from KKT: (2, 1); obj = 4 + 4 + 2 - 12 - 8 = -10.
TEST(lp_parser_solve, qp_with_cross_term)
{
  expect_optimal_solution(R"LP(
Minimize
  obj: -6 x1 - 8 x2 + [ 2 x1 ^ 2 + 4 x1 * x2 + 4 x2 ^ 2 ] / 2
Subject To
  c1: x1 + x2 <= 10
Bounds
  -100 <= x1 <= 100
  -100 <= x2 <= 100
End
)LP",
                          -10.0,
                          {2.0, 1.0});
}

// Maximizing a concave quadratic is converted to minimizing its negation.
// The objective is the negation of qp_with_cross_term plus a constant:
// maximize -x1^2 - 2 x1 x2 - 2 x2^2 + 6 x1 + 8 x2 + 5.
TEST(lp_parser_solve, qp_maximize_concave)
{
  expect_optimal_solution(R"LP(
Maximize
  obj: 5 + 6 x1 + 8 x2 + [ -2 x1 ^ 2 - 4 x1 * x2 - 4 x2 ^ 2 ] / 2
Subject To
  c1: x1 + x2 <= 10
Bounds
  -100 <= x1 <= 100
  -100 <= x2 <= 100
End
)LP",
                          15.0,
                          {2.0, 1.0});
}

// Maximization QP with dual / reduced-cost check.
// maximize 4 x1 + x2 - 0.5 (x1^2 + x2^2) s.t. x1 + x2 = 1, x >= 0
// Optimal: x = (1, 0), obj = 3.5.
// Duals satisfy A^T y + z = c + Q x on the user's objective, so with
// c + Q x = (3, 1) and A^T y = (3, 3): y = 3 and z = (0, -2).
TEST(lp_parser_solve, qp_maximize_duals)
{
  raft::handle_t handle;
  auto problem  = io::read_lp_from_string<int, double>(R"LP(
Maximize
  obj: 4 x1 + x2 + [ - x1 ^ 2 - x2 ^ 2 ] / 2
Subject To
  eq1: x1 + x2 = 1
Bounds
  x1 >= 0
  x2 >= 0
End
)LP");
  auto settings = pdlp_solver_settings_t<int, double>();
  auto solution = solve_lp(&handle, problem, settings);

  ASSERT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(solution.get_objective_value(), 3.5, 1e-4);

  auto stream = handle.get_stream();
  auto h_x    = cuopt::host_copy(solution.get_primal_solution(), stream);
  auto h_y    = cuopt::host_copy(solution.get_dual_solution(), stream);
  auto h_z    = cuopt::host_copy(solution.get_reduced_cost(), stream);

  ASSERT_EQ(h_x.size(), 2u);
  ASSERT_EQ(h_y.size(), 1u);
  ASSERT_EQ(h_z.size(), 2u);
  EXPECT_NEAR(h_x[0], 1.0, 1e-4);
  EXPECT_NEAR(h_x[1], 0.0, 1e-4);
  EXPECT_NEAR(h_y[0], 3.0, 1e-4);
  EXPECT_NEAR(h_z[0], 0.0, 1e-4);
  EXPECT_NEAR(h_z[1], -2.0, 1e-4);
}

// Maximize with a nonzero objective offset, quadratic constraints, and no
// quadratic objective. Pins that obj_constant is negated for every maximize
// problem in cuopt_optimization_problem_to_user_problem, not only when Q is
// nonempty. Optimal: x = 1, objective = 5 + 1 = 6.
TEST(lp_parser_solve, qcqp_maximize_offset_no_q_objective)
{
  raft::handle_t handle;
  auto problem = io::read_lp_from_string<int, double>(R"LP(
Maximize
  obj: 5 + x
Subject To
  ball: [ x ^ 2 ] <= 1
Bounds
  x free
End
)LP");
  ASSERT_FALSE(problem.has_quadratic_objective());
  ASSERT_TRUE(problem.has_quadratic_constraints());

  auto settings = pdlp_solver_settings_t<int, double>();
  auto solution = solve_lp(&handle, problem, settings);

  ASSERT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(solution.get_objective_value(), 6.0, 1e-4);

  auto h_x = cuopt::host_copy(solution.get_primal_solution(), handle.get_stream());
  ASSERT_EQ(h_x.size(), 1u);
  EXPECT_NEAR(h_x[0], 1.0, 1e-4);
}

// Quadratic objective with a negative cross-term coefficient. This
// exercises the same upper-triangular off-diagonal storage path with a
// sign that gets carried through parse_quadratic_bracket via the per-term
// sign of `- 4 x1 * x2`.
//
// Minimize x1^2 - 2 x1 x2 + 2 x2^2 - 4 x1 s.t. x1 + x2 <= 100.
// Hessian H = [[2, -2], [-2, 4]] is positive definite.
// Unconstrained optimum from KKT: (4, 2); obj = 16 - 16 + 8 - 16 = -8.
TEST(lp_parser_solve, qp_with_negative_cross_term)
{
  expect_optimal_solution(R"LP(
Minimize
  obj: -4 x1 + [ 2 x1 ^ 2 - 4 x1 * x2 + 4 x2 ^ 2 ] / 2
Subject To
  c1: x1 + x2 <= 100
Bounds
  -100 <= x1 <= 100
  -100 <= x2 <= 100
End
)LP",
                          -8.0,
                          {4.0, 2.0});
}

// Dual residual check for QP.
TEST(lp_parser_solve, qp_diagonal_only_dual_residual)
{
  raft::handle_t handle;
  auto problem  = io::read_lp_from_string<int, double>(R"LP(
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

  ASSERT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_NEAR(solution.get_objective_value(), -32.0, 1e-4);
  EXPECT_NEAR(solution.get_additional_termination_information().l2_dual_residual, 0.0, 1e-4);
}

}  // namespace cuopt::mathematical_optimization
