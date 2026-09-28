/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/primal.hpp>
#include <dual_simplex/solve.hpp>
#include <math_optimization/tic_toc.hpp>
#include <utilities/logger.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace cuopt::mathematical_optimization::simplex::test {

class primal_simplex : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // min -3*x - 2*y, x + y <= 4, 2*x + y <= 5, 0 <= x,y <= 10.
    // The last two columns are slacks; every solve starts from a fresh slack basis.
    lp.A.col_start           = {0, 2, 4, 5, 6};
    lp.A.i                   = {0, 1, 0, 1, 0, 1};
    lp.A.x                   = {1.0, 2.0, 1.0, 1.0, 1.0, 1.0};
    lp.objective             = {-3.0, -2.0, 0.0, 0.0};
    lp.rhs                   = {4.0, 5.0};
    lp.lower                 = {0.0, 0.0, 0.0, 0.0};
    lp.upper                 = {10.0, 10.0, inf, inf};
    lp.obj_scale             = 1.0;
    settings.iteration_limit = 100;
  }

  void expect_optimum(const std::vector<double>& expected_x, double expected_objective)
  {
    ASSERT_EQ(solution.x.size(), expected_x.size());
    std::vector<double> residual = lp.rhs;
    double objective             = 0.0;
    for (int j = 0; j < lp.num_cols; ++j) {
      EXPECT_NEAR(solution.x[j], expected_x[j], 1e-6) << "column " << j;
      EXPECT_GE(solution.x[j], lp.lower[j] - 1e-6) << "column " << j;
      EXPECT_LE(solution.x[j], lp.upper[j] + 1e-6) << "column " << j;
      objective += lp.objective[j] * solution.x[j];
      for (int k = lp.A.col_start[j]; k < lp.A.col_start[j + 1]; ++k) {
        residual[lp.A.i[k]] -= lp.A.x[k] * solution.x[j];
      }
    }
    EXPECT_NEAR(objective, expected_objective, 1e-6);
    EXPECT_NEAR(solution.objective, expected_objective, 1e-6);
    EXPECT_NEAR(solution.user_objective, expected_objective, 1e-6);
    for (int i = 0; i < lp.num_rows; ++i) {
      EXPECT_NEAR(residual[i], 0.0, 1e-6) << "row " << i;
    }
  }

  cuopt::init_logger_t log{"", true};
  raft::handle_t handle{};
  lp_problem_t<int, double> lp{&handle, 2, 4, 6};
  simplex_solver_settings_t<int, double> settings;
  lp_solution_t<int, double> solution{2, 4};
  std::vector<variable_status_t> vstatus{variable_status_t::NONBASIC_LOWER,
                                         variable_status_t::NONBASIC_LOWER,
                                         variable_status_t::BASIC,
                                         variable_status_t::BASIC};
  int iter = 0;
};

TEST_F(primal_simplex, bounded_phase2)
{
  ASSERT_EQ(primal_phase2(2, tic(), lp, settings, vstatus, solution, iter),
            primal_status_t::OPTIMAL);
  EXPECT_GT(iter, 0);
  expect_optimum({1.0, 3.0, 0.0, 0.0}, -9.0);

  // Exercise conversion, presolve, scaling and postsolve through the user entry point too.
  user_problem_t<int, double> user_problem(&handle);
  user_problem.num_rows = 2;
  user_problem.num_cols = 2;
  user_problem.A.resize(2, 2, 4);
  user_problem.A.col_start    = {0, 2, 4};
  user_problem.A.i            = {0, 1, 0, 1};
  user_problem.A.x            = {1.0, 2.0, 1.0, 1.0};
  user_problem.objective      = {-3.0, -2.0};
  user_problem.rhs            = {4.0, 5.0};
  user_problem.row_sense      = {'L', 'L'};
  user_problem.lower          = {0.0, 0.0};
  user_problem.upper          = {10.0, 10.0};
  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "primal_bounded_phase2";
  user_problem.row_names      = {"capacity", "resource"};
  user_problem.col_names      = {"x", "y"};
  user_problem.var_types      = {variable_type_t::CONTINUOUS, variable_type_t::CONTINUOUS};
  settings.dualize            = 0;
  lp_solution_t<int, double> user_solution(2, 2);
  ASSERT_EQ(solve_linear_program_with_primal(user_problem, settings, tic(), user_solution),
            lp_status_t::OPTIMAL);
  ASSERT_EQ(user_solution.x.size(), 2);
  EXPECT_NEAR(user_solution.x[0], 1.0, 1e-6);
  EXPECT_NEAR(user_solution.x[1], 3.0, 1e-6);
  EXPECT_NEAR(user_solution.objective, -9.0, 1e-6);
  EXPECT_NEAR(user_solution.user_objective, -9.0, 1e-6);
  EXPECT_NEAR(-3.0 * user_solution.x[0] - 2.0 * user_solution.x[1], -9.0, 1e-6);
  EXPECT_NEAR(user_solution.x[0] + user_solution.x[1] - 4.0, 0.0, 1e-6);
  EXPECT_NEAR(2.0 * user_solution.x[0] + user_solution.x[1] - 5.0, 0.0, 1e-6);
}

TEST_F(primal_simplex, phase1_recovery)
{
  // x + y >= 2 gives an infeasible initial slack s = -2.
  // Phase I must recover feasibility, then Phase II minimizes the original objective.
  lp.A.x[0] = -1.0;
  lp.A.x[2] = -1.0;
  lp.rhs[0] = -2.0;
  ASSERT_EQ(primal_phase2(2, tic(), lp, settings, vstatus, solution, iter),
            primal_status_t::OPTIMAL);
  EXPECT_GT(iter, 0);
  expect_optimum({0.0, 5.0, 3.0, 0.0}, -10.0);
}

TEST_F(primal_simplex, infeasible)
{
  // x + y >= 6 contradicts 2*x + y <= 5 for nonnegative x,y.
  lp.A.x[0] = -1.0;
  lp.A.x[2] = -1.0;
  lp.rhs[0] = -6.0;
  ASSERT_EQ(primal_phase2(2, tic(), lp, settings, vstatus, solution, iter),
            primal_status_t::PRIMAL_INFEASIBLE);
}

TEST_F(primal_simplex, unbounded)
{
  // -x - y <= 4 and -2*x - y <= 5 permit x to increase without limit.
  lp.A.x   = {-1.0, -2.0, -1.0, -1.0, 1.0, 1.0};
  lp.upper = {inf, inf, inf, inf};
  ASSERT_EQ(primal_phase2(2, tic(), lp, settings, vstatus, solution, iter),
            primal_status_t::PRIMAL_UNBOUNDED);
}

TEST_F(primal_simplex, termination_limits)
{
  // Each limit gets its own fresh solution and slack basis, with no sleeps needed.
  for (bool expired_time : {false, true}) {
    SCOPED_TRACE(expired_time ? "expired time limit" : "zero iteration limit");
    lp_solution_t<int, double> limited_solution(lp.num_rows, lp.num_cols);
    auto limited_vstatus     = vstatus;
    int limited_iter         = 0;
    settings.iteration_limit = expired_time ? 100 : 0;
    settings.time_limit      = expired_time ? 1.0 : inf;
    const double start_time  = tic() - (expired_time ? 2.0 : 0.0);
    EXPECT_EQ(
      primal_phase2(2, start_time, lp, settings, limited_vstatus, limited_solution, limited_iter),
      expired_time ? primal_status_t::TIME_LIMIT : primal_status_t::ITERATION_LIMIT);
    EXPECT_EQ(limited_iter, 0);
  }
}

}  // namespace cuopt::mathematical_optimization::simplex::test
