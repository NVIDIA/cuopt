/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>

#include <utilities/common_utils.hpp>

#include <gtest/gtest.h>

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/user_problem.hpp>

#include <mps_parser/parser.hpp>

namespace cuopt::linear_programming::dual_simplex::test {

TEST(dual_simplex, chess_set)
{
  namespace dual_simplex = cuopt::linear_programming::dual_simplex;
  dual_simplex::user_problem_t<int, double> user_problem;
  // maximize   5*xs + 20*xl
  // subject to  1*xs +  3*xl <= 200
  //             3*xs +  2*xl <= 160
  constexpr int m  = 2;
  constexpr int n  = 2;
  constexpr int nz = 4;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.resize(n);
  user_problem.objective[0] = -5;
  user_problem.objective[1] = -20;
  user_problem.A.m          = m;
  user_problem.A.n          = n;
  user_problem.A.nz_max     = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start.resize(n + 1);
  user_problem.A.col_start[0] = 0;
  user_problem.A.col_start[1] = 2;
  user_problem.A.col_start[2] = 4;
  user_problem.A.i[0]         = 0;
  user_problem.A.x[0]         = 1.0;
  user_problem.A.i[1]         = 1;
  user_problem.A.x[1]         = 3.0;
  user_problem.A.i[2]         = 0;
  user_problem.A.x[2]         = 3.0;
  user_problem.A.i[3]         = 1;
  user_problem.A.x[3]         = 2.0;
  user_problem.rhs.resize(m);
  user_problem.rhs[0] = 200;
  user_problem.rhs[1] = 160;
  user_problem.row_sense.resize(m);
  user_problem.row_sense[0] = 'L';
  user_problem.row_sense[1] = 'L';
  user_problem.lower.resize(n);
  user_problem.lower[0] = 0;
  user_problem.lower[1] = 0.0;
  user_problem.upper.resize(n);
  user_problem.upper[0]       = dual_simplex::inf;
  user_problem.upper[1]       = dual_simplex::inf;
  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "chess set";
  user_problem.row_names.resize(m);
  user_problem.row_names[0] = "boxwood";
  user_problem.row_names[1] = "lathe hours";
  user_problem.col_names.resize(n);
  user_problem.col_names[0] = "xs";
  user_problem.col_names[1] = "xl";
  user_problem.obj_constant = 0.0;
  user_problem.var_types.resize(n);
  user_problem.var_types[0] = dual_simplex::variable_type_t::CONTINUOUS;
  user_problem.var_types[1] = dual_simplex::variable_type_t::CONTINUOUS;

  dual_simplex::simplex_solver_settings_t<int, double> settings;
  dual_simplex::lp_solution_t<int, double> solution(user_problem.num_rows, user_problem.num_cols);
  EXPECT_EQ((dual_simplex::solve_linear_program(user_problem, settings, solution)),
            dual_simplex::lp_status_t::OPTIMAL);
  const double objective = -solution.objective;
  EXPECT_NEAR(objective, 1333.33, 1e-2);
  EXPECT_NEAR(solution.x[0], 0.0, 1e-6);
  EXPECT_NEAR(solution.x[1], 66.6667, 1e-3);

  user_problem.var_types[0] = dual_simplex::variable_type_t::INTEGER;
  user_problem.var_types[1] = dual_simplex::variable_type_t::INTEGER;

  EXPECT_EQ((dual_simplex::solve(user_problem, settings, solution.x)), 0);
}

TEST(dual_simplex, burglar)
{
  constexpr int num_items     = 8;
  constexpr double max_weight = 102;

  std::vector<double> value({15, 100, 90, 60, 40, 15, 10, 1});
  std::vector<double> weight({2, 20, 20, 30, 40, 30, 60, 10});

  // maximize  sum_i value[i] * take[i]
  //           sum_i weight[i] * take[i] <= max_weight
  //           take[i] binary for all i

  cuopt::linear_programming::dual_simplex::user_problem_t<int, double> user_problem;
  constexpr int m  = 1;
  constexpr int n  = num_items;
  constexpr int nz = num_items;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.objective[j] = -value[j];
  }
  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start.resize(n + 1);
  for (int j = 0; j < num_items; ++j) {
    user_problem.A.col_start[j] = j;
    user_problem.A.i[j]         = 0;
    user_problem.A.x[j]         = weight[j];
  }
  user_problem.A.col_start[n] = nz;
  user_problem.rhs.resize(m);
  user_problem.rhs[0] = max_weight;
  user_problem.row_sense.resize(m);
  user_problem.row_sense[0] = 'L';
  user_problem.lower.resize(n);
  user_problem.upper.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.lower[j] = 0.0;
    user_problem.upper[j] = 1.0;
  }
  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "burglar";
  user_problem.row_names.resize(m);
  user_problem.row_names[0] = "weight restriction";
  user_problem.col_names.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.col_names[j] = "x";
  }
  user_problem.obj_constant = 0.0;
  user_problem.var_types.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.var_types[j] = cuopt::linear_programming::dual_simplex::variable_type_t::INTEGER;
  }
  cuopt::linear_programming::dual_simplex::simplex_solver_settings_t<int, double> settings;
  std::vector<double> solution(num_items);
  EXPECT_EQ((cuopt::linear_programming::dual_simplex::solve(user_problem, settings, solution)), 0);
  double objective = 0.0;
  for (int j = 0; j < num_items; ++j) {
    objective += value[j] * solution[j];
  }
  EXPECT_NEAR(objective, 280, 1e-6);
  EXPECT_NEAR(solution[0], 1, 1e-6);
  EXPECT_NEAR(solution[1], 1, 1e-6);
  EXPECT_NEAR(solution[2], 1, 1e-6);
  EXPECT_NEAR(solution[3], 1, 1e-6);
  EXPECT_NEAR(solution[5], 1, 1e-6);
}

TEST(dual_simplex, empty_columns)
{
  // Same as burglar problem above but with an empty column inserted
  constexpr int num_items     = 9;
  constexpr double max_weight = 102;

  std::vector<double> value({15, 100, 90, 0, 60, 40, 15, 10, 1});
  std::vector<double> weight({2, 20, 20, 0, 30, 40, 30, 60, 10});

  // maximize  sum_i value[i] * take[i]
  //           sum_i weight[i] * take[i] <= max_weight
  //           take[i] binary for all i

  cuopt::linear_programming::dual_simplex::user_problem_t<int, double> user_problem;
  constexpr int m  = 1;
  constexpr int n  = num_items;
  constexpr int nz = num_items - 1;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.objective[j] = -value[j];
  }
  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start.resize(n + 1);
  int nnz = 0;
  for (int j = 0; j < num_items; ++j) {
    user_problem.A.col_start[j] = nnz;
    if (weight[j] > 0) {
      user_problem.A.i[nnz] = 0;
      user_problem.A.x[nnz] = weight[j];
      nnz++;
    }
  }
  user_problem.A.col_start[n] = nnz;
  user_problem.rhs.resize(m);
  user_problem.rhs[0] = max_weight;
  user_problem.row_sense.resize(m);
  user_problem.row_sense[0] = 'L';
  user_problem.lower.resize(n);
  user_problem.upper.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.lower[j] = 0.0;
    user_problem.upper[j] = 1.0;
  }
  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "burglar";
  user_problem.row_names.resize(m);
  user_problem.row_names[0] = "weight restriction";
  user_problem.col_names.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.col_names[j] = "x";
  }
  user_problem.obj_constant = 0.0;
  user_problem.var_types.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.var_types[j] =
      cuopt::linear_programming::dual_simplex::variable_type_t::CONTINUOUS;
  }

  cuopt::linear_programming::dual_simplex::simplex_solver_settings_t<int, double> settings;

  cuopt::linear_programming::dual_simplex::lp_solution_t<int, double> solution(
    user_problem.num_rows, user_problem.num_cols);
  EXPECT_EQ((cuopt::linear_programming::dual_simplex::solve_linear_program(
              user_problem, settings, solution)),
            cuopt::linear_programming::dual_simplex::lp_status_t::OPTIMAL);
  double objective = 0.0;
  for (int j = 0; j < num_items; ++j) {
    objective += value[j] * solution.x[j];
  }
  EXPECT_NEAR(objective, 295, 1e-6);
  EXPECT_NEAR(solution.x[0], 1, 1e-6);
  EXPECT_NEAR(solution.x[1], 1, 1e-6);
  EXPECT_NEAR(solution.x[2], 1, 1e-6);
  EXPECT_NEAR(solution.x[3], 0, 1e-6);
  EXPECT_NEAR(solution.x[4], 1, 1e-6);
  EXPECT_NEAR(solution.x[5], 0.75, 1e-6);
  EXPECT_NEAR(solution.x[6], 0, 1e-6);
  EXPECT_NEAR(solution.x[7], 0, 1e-6);
  EXPECT_NEAR(solution.x[8], 0, 1e-6);
}

TEST(dual_simplex, dual_variable_greater_than)
{
  // minimize   3*x0 + 2 * x1
  // subject to  x0 + x1  >= 1
  //             x0 + 2x1 >= 3
  //             x0, x1 >= 0

  cuopt::linear_programming::dual_simplex::user_problem_t<int, double> user_problem;
  constexpr int m  = 2;
  constexpr int n  = 2;
  constexpr int nz = 4;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.resize(n);
  user_problem.objective[0] = 3.0;
  user_problem.objective[1] = 2.0;
  user_problem.A.m          = m;
  user_problem.A.n          = n;
  user_problem.A.nz_max     = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start.resize(n + 1);
  user_problem.A.col_start[0] = 0;  // x0 start
  user_problem.A.col_start[1] = 2;
  user_problem.A.col_start[2] = 4;

  int nnz                 = 0;
  user_problem.A.i[nnz]   = 0;
  user_problem.A.x[nnz++] = 1.0;
  user_problem.A.i[nnz]   = 1;
  user_problem.A.x[nnz++] = 1.0;
  user_problem.A.i[nnz]   = 0;
  user_problem.A.x[nnz++] = 1.0;
  user_problem.A.i[nnz]   = 1;
  user_problem.A.x[nnz++] = 2.0;
  user_problem.A.print_matrix();
  EXPECT_EQ(nnz, nz);

  user_problem.rhs.resize(m);
  user_problem.rhs[0] = 1.0;
  user_problem.rhs[1] = 3.0;

  user_problem.row_sense.resize(m);
  user_problem.row_sense[0] = 'G';
  user_problem.row_sense[1] = 'G';

  user_problem.lower.resize(n);
  user_problem.lower[0] = 0.0;
  user_problem.lower[1] = 0.0;

  user_problem.upper.resize(n);
  user_problem.upper[0] = dual_simplex::inf;
  user_problem.upper[1] = dual_simplex::inf;

  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "dual_variable_greater_than";

  dual_simplex::simplex_solver_settings_t<int, double> settings;
  dual_simplex::lp_solution_t<int, double> solution(user_problem.num_rows, user_problem.num_cols);
  EXPECT_EQ((dual_simplex::solve_linear_program(user_problem, settings, solution)),
            dual_simplex::lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 3.0, 1e-6);
  EXPECT_NEAR(solution.x[0], 0.0, 1e-6);
  EXPECT_NEAR(solution.x[1], 1.5, 1e-6);
  EXPECT_NEAR(solution.y[0], 0.0, 1e-6);
  EXPECT_NEAR(solution.y[1], 1.0, 1e-6);
  EXPECT_NEAR(solution.z[0], 2.0, 1e-6);
  EXPECT_NEAR(solution.z[1], 0.0, 1e-6);
}

}  // namespace cuopt::linear_programming::dual_simplex::test
