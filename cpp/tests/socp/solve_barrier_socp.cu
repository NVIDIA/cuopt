/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/scaling.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/user_problem.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>

#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::simplex::test {

// This serves as both a warm up but also a mandatory initial call to setup cuSparse and cuBLAS
static void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

TEST(barrier, cone_metadata_reindexed_when_slack_is_inserted_before_cones)
{
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m       = 1;
  constexpr int n       = 5;
  constexpr int nz      = 5;
  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.assign(n, 0.0);
  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start.resize(n + 1);
  for (int j = 0; j < n; ++j) {
    user_problem.A.col_start[j] = j;
    user_problem.A.i[j]         = 0;
    user_problem.A.x[j]         = 1.0;
  }
  user_problem.A.col_start[n] = nz;
  user_problem.rhs            = {1.0};
  user_problem.row_sense      = {'L'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);
  user_problem.num_range_rows         = 0;
  user_problem.second_order_cone_dims = {2, 2};
  user_problem.cone_var_start         = 1;

  simplex_solver_settings_t<int, double> settings;
  settings.barrier       = true;
  settings.dualize       = 0;
  settings.scale_columns = false;

  std::vector<int> new_slacks;
  dualize_info_t<int, double> dualize_info;
  lp_problem_t<int, double> original_lp(user_problem.handle_ptr, 1, 1, 1);
  convert_user_problem(user_problem, settings, original_lp, new_slacks, dualize_info);

  ASSERT_EQ(new_slacks.size(), 1);
  EXPECT_EQ(new_slacks[0], 1);
  EXPECT_EQ(original_lp.num_cols, 6);
  EXPECT_EQ(original_lp.second_order_cone_dims, user_problem.second_order_cone_dims);
  EXPECT_EQ(original_lp.cone_var_start, 2);

  lp_problem_t<int, double> barrier_lp(user_problem.handle_ptr,
                                       original_lp.num_rows,
                                       original_lp.num_cols,
                                       original_lp.A.col_start[original_lp.num_cols]);
  std::vector<double> column_scales;
  std::vector<double> row_scales;
  scaling(original_lp, settings, barrier_lp, column_scales, row_scales);

  EXPECT_EQ(barrier_lp.second_order_cone_dims, user_problem.second_order_cone_dims);
  EXPECT_EQ(barrier_lp.cone_var_start, 2);
}

TEST(barrier, presolve_reindexes_cone_start_after_empty_column_removal)
{
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 4;
  constexpr int nz = 3;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 0, 1, 2, 3};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = -1.0;
  user_problem.A.i[2]      = 0;
  user_problem.A.x[2]      = 0.5;

  user_problem.rhs       = {1.0};
  user_problem.row_sense = {'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);
  user_problem.num_range_rows         = 0;
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = false;

  std::vector<int> new_slacks;
  dualize_info_t<int, double> dualize_info;
  lp_problem_t<int, double> original_lp(user_problem.handle_ptr, 1, 1, 1);
  convert_user_problem(user_problem, settings, original_lp, new_slacks, dualize_info);

  presolve_info_t<int, double> presolve_info;
  lp_problem_t<int, double> presolved_lp(user_problem.handle_ptr, 1, 1, 1);
  ASSERT_EQ(presolve(original_lp, settings, presolved_lp, presolve_info), 0);

  EXPECT_EQ(presolved_lp.num_cols, 3);
  EXPECT_EQ(presolved_lp.second_order_cone_dims, std::vector<int>({3}));
  EXPECT_EQ(presolved_lp.cone_var_start, 0);

  lp_problem_t<int, double> barrier_lp(user_problem.handle_ptr,
                                       presolved_lp.num_rows,
                                       presolved_lp.num_cols,
                                       presolved_lp.A.col_start[presolved_lp.num_cols]);
  std::vector<double> column_scales;
  std::vector<double> row_scales;
  ASSERT_EQ(scaling(presolved_lp, settings, barrier_lp, column_scales, row_scales), 0);
  EXPECT_EQ(barrier_lp.cone_var_start, 0);
}

TEST(barrier, presolve_keeps_direct_free_variables_before_cones)
{
  // Layout: [x0, x1 | cone x2, x3, x4] with x0, x1 free and a 3-dimensional SOC block.
  // SOCP barrier presolve keeps direct free variables (no x = v - w split); cone_var_start
  // and column count stay unchanged.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 5;
  constexpr int nz = 5;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 0.0, 0.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2, 3, 4, 5};
  for (int j = 0; j < n; ++j) {
    user_problem.A.i[j] = 0;
    user_problem.A.x[j] = 1.0;
  }

  user_problem.rhs       = {1.0};
  user_problem.row_sense = {'E'};
  user_problem.lower     = {-inf, -inf, 0.0, 0.0, 0.0};
  user_problem.upper.assign(n, inf);
  user_problem.num_range_rows         = 0;
  user_problem.cone_var_start         = 2;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = false;

  std::vector<int> new_slacks;
  dualize_info_t<int, double> dualize_info;
  lp_problem_t<int, double> original_lp(user_problem.handle_ptr, 1, 1, 1);
  convert_user_problem(user_problem, settings, original_lp, new_slacks, dualize_info);

  presolve_info_t<int, double> presolve_info;
  lp_problem_t<int, double> presolved_lp(user_problem.handle_ptr, 1, 1, 1);
  ASSERT_EQ(presolve(original_lp, settings, presolved_lp, presolve_info), 0);

  EXPECT_EQ(presolved_lp.num_cols, 5);
  EXPECT_EQ(presolved_lp.cone_var_start, 2);
  EXPECT_EQ(presolved_lp.second_order_cone_dims, std::vector<int>({3}));
  EXPECT_TRUE(presolve_info.free_variable_pairs.empty());
  ASSERT_EQ(presolve_info.direct_free_variables.size(), 2);
  EXPECT_EQ(presolve_info.direct_free_variables[0], 0);
  EXPECT_EQ(presolve_info.direct_free_variables[1], 1);
  EXPECT_EQ(presolved_lp.lower[0], -inf);
  EXPECT_EQ(presolved_lp.lower[1], -inf);
  EXPECT_EQ(presolved_lp.upper[0], inf);
  EXPECT_EQ(presolved_lp.upper[1], inf);
}

TEST(barrier, rejects_middle_cone_input_before_barrier)
{
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 3;
  constexpr int n  = 5;
  constexpr int nz = 3;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0, 0.0, 0.0, 1.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 1, 2, 2, 3};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 1;
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 2;
  user_problem.A.x[2]      = 1.0;

  user_problem.rhs       = {2.0, 1.0, 3.0};
  user_problem.row_sense = {'E', 'E', 'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);
  user_problem.num_range_rows         = 0;
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier = true;
  settings.dualize = 0;
  lp_solution_t<int, double> solution(m, n);

  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);
  EXPECT_EQ(status, lp_status_t::NUMERICAL_ISSUES);
}

TEST(barrier, socp_min_x0_subject_to_norm_constraint)
{
  // minimize x_0
  // subject to x_1 = 1
  //            (x_0, x_1, x_2) in Q^3
  //
  // Optimal: x* = (1, 1, 0), obj* = 1

  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 3;
  constexpr int nz = 1;

  user_problem.num_rows = m;
  user_problem.num_cols = n;

  user_problem.objective = {1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 0, 1, 1};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;

  user_problem.rhs       = {1.0};
  user_problem.row_sense = {'E'};

  user_problem.lower = {0.0, 0.0, 0.0};
  user_problem.upper = {inf, inf, inf};

  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "socp_norm_cone";

  user_problem.cone_var_start         = 0;
  user_problem.second_order_cone_dims = {3};

  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);
  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[2]), 0.0, 1e-4);
}

TEST(barrier, mixed_linear_and_soc_block)
{
  // Variables ordered as [l | t, u, v], where (t, u, v) \in Q^3.
  //
  // minimize   l
  // subject to l - t = 0
  //            u     = 1
  //            (t, u, v) in Q^3
  //
  // Optimal: l* = 1, t* = 1, u* = 1, v* = 0, obj* = 1.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 4;
  constexpr int nz = 4;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l, t, u, v
  user_problem.A.col_start = {0, 1, 2, 3, 3};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = -1.0;
  user_problem.A.i[2]      = 1;
  user_problem.A.x[2]      = 1.0;

  user_problem.rhs       = {0.0, 1.0};
  user_problem.row_sense = {'E', 'E'};

  user_problem.lower = {0.0, 0.0, 0.0, 0.0};
  user_problem.upper = {inf, inf, inf, inf};

  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "mixed_linear_and_soc_block";

  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[3]), 0.0, 1e-4);
}

TEST(barrier, mixed_linear_and_soc_tail_coupling)
{
  // Variables ordered as [l | t, u, v], where (t, u, v) \in Q^3.
  //
  // minimize   t
  // subject to l - u = 0
  //            l + u = 2
  //            (t, u, v) in Q^3
  //
  // Optimal: l* = 1, t* = 1, u* = 1, v* = 0, obj* = 1.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 4;
  constexpr int nz = 4;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l, t, u, v
  user_problem.A.col_start = {0, 2, 2, 4, 4};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 1;
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 0;
  user_problem.A.x[2]      = -1.0;
  user_problem.A.i[3]      = 1;
  user_problem.A.x[3]      = 1.0;

  user_problem.rhs       = {0.0, 2.0};
  user_problem.row_sense = {'E', 'E'};
  user_problem.lower     = {0.0, 0.0, 0.0, 0.0};
  user_problem.upper     = {inf, inf, inf, inf};

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "mixed_linear_and_soc_tail_coupling";
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = true;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[3]), 0.0, 1e-4);
}

TEST(barrier, mixed_linear_and_soc_tail_coupling_with_inequality)
{
  // Variables ordered as [l | t, u, v], where (t, u, v) \in Q^3.
  //
  // minimize   t
  // subject to l - u = 0
  //            l + u >= 2
  //            (t, u, v) in Q^3
  //
  // Optimal: l* = 1, t* = 1, u* = 1, v* = 0, obj* = 1.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 4;
  constexpr int nz = 4;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l, t, u, v
  user_problem.A.col_start = {0, 2, 2, 4, 4};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 1;
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 0;
  user_problem.A.x[2]      = -1.0;
  user_problem.A.i[3]      = 1;
  user_problem.A.x[3]      = 1.0;

  user_problem.rhs       = {0.0, 2.0};
  user_problem.row_sense = {'E', 'G'};
  user_problem.lower     = {0.0, 0.0, 0.0, 0.0};
  user_problem.upper     = {inf, inf, inf, inf};

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "mixed_linear_and_soc_tail_coupling_with_inequality";
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = true;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[3]), 0.0, 1e-4);
}

TEST(barrier, mixed_linear_and_two_soc_blocks)
{
  // Variables ordered as [l1, l2 | t1, u1, v1 | t2, u2, v2],
  // where (t1, u1, v1), (t2, u2, v2) \in Q^3.
  //
  // minimize   t1 + t2
  // subject to l1 - u1 = 0
  //            l2 - u2 = 0
  //            l1 + l2 = 3
  //            l1 - l2 = 1
  //
  // Optimal: l1* = 2, l2* = 1, t1* = 2, u1* = 2, v1* = 0,
  //          t2* = 1, u2* = 1, v2* = 0, obj* = 3.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 4;
  constexpr int n  = 8;
  constexpr int nz = 8;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l1, l2, t1, u1, v1, t2, u2, v2
  user_problem.A.col_start = {0, 3, 6, 6, 7, 7, 7, 8, 8};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 2;
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 3;
  user_problem.A.x[2]      = 1.0;
  user_problem.A.i[3]      = 1;
  user_problem.A.x[3]      = 1.0;
  user_problem.A.i[4]      = 2;
  user_problem.A.x[4]      = 1.0;
  user_problem.A.i[5]      = 3;
  user_problem.A.x[5]      = -1.0;
  user_problem.A.i[6]      = 0;
  user_problem.A.x[6]      = -1.0;
  user_problem.A.i[7]      = 1;
  user_problem.A.x[7]      = -1.0;

  user_problem.rhs       = {0.0, 0.0, 3.0, 1.0};
  user_problem.row_sense = {'E', 'E', 'E', 'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "mixed_linear_and_two_soc_blocks";
  user_problem.cone_var_start         = 2;
  user_problem.second_order_cone_dims = {3, 3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 3.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[3], 2.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[4]), 0.0, 1e-4);
  EXPECT_NEAR(solution.x[5], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[6], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[7]), 0.0, 1e-4);
}

TEST(barrier, mixed_linear_and_two_soc_blocks_with_inequality)
{
  // Variables ordered as [l1, l2 | t1, u1, v1 | t2, u2, v2],
  // where (t1, u1, v1), (t2, u2, v2) \in Q^3.
  //
  // minimize   t1 + t2
  // subject to l1 - u1 = 0
  //            l2 - u2 = 0
  //            l1 + l2 >= 3
  //            l1 - l2 = 1
  //
  // Optimal: l1* = 2, l2* = 1, t1* = 2, u1* = 2, v1* = 0,
  //          t2* = 1, u2* = 1, v2* = 0, obj* = 3.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 4;
  constexpr int n  = 8;
  constexpr int nz = 8;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l1, l2, t1, u1, v1, t2, u2, v2
  user_problem.A.col_start = {0, 3, 6, 6, 7, 7, 7, 8, 8};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 2;
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 3;
  user_problem.A.x[2]      = 1.0;
  user_problem.A.i[3]      = 1;
  user_problem.A.x[3]      = 1.0;
  user_problem.A.i[4]      = 2;
  user_problem.A.x[4]      = 1.0;
  user_problem.A.i[5]      = 3;
  user_problem.A.x[5]      = -1.0;
  user_problem.A.i[6]      = 0;
  user_problem.A.x[6]      = -1.0;
  user_problem.A.i[7]      = 1;
  user_problem.A.x[7]      = -1.0;

  user_problem.rhs       = {0.0, 0.0, 3.0, 1.0};
  user_problem.row_sense = {'E', 'E', 'G', 'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "mixed_linear_and_two_soc_blocks_with_inequality";
  user_problem.cone_var_start         = 2;
  user_problem.second_order_cone_dims = {3, 3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  settings.scale_columns    = true;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 3.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[3], 2.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[4]), 0.0, 1e-4);
  EXPECT_NEAR(solution.x[5], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[6], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[7]), 0.0, 1e-4);
}

TEST(barrier, free_linear_prefix_is_uncrushed_correctly_with_soc_block)
{
  // Variables ordered as [l | t, u, v], where (t, u, v) \in Q^3 and l is free.
  //
  // minimize   t
  // subject to l - u = 0
  //            u     = 1
  //            (t, u, v) in Q^3
  //
  // Direct free variable l is kept through presolve; end-to-end solve returns
  // l* = 1, t* = 1, u* = 1, v* = 0, obj* = 1.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 4;
  constexpr int nz = 3;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l, t, u, v
  user_problem.A.col_start = {0, 1, 1, 3, 3};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = -1.0;
  user_problem.A.i[2]      = 1;
  user_problem.A.x[2]      = 1.0;

  user_problem.rhs       = {0.0, 1.0};
  user_problem.row_sense = {'E', 'E'};
  user_problem.lower     = {-inf, 0.0, 0.0, 0.0};
  user_problem.upper     = {inf, inf, inf, inf};

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "free_linear_prefix_is_uncrushed_correctly_with_soc_block";
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[3]), 0.0, 1e-4);
}

TEST(barrier, qp_with_soc_block)
{
  // Variables ordered as [l | t, u, v], where (t, u, v) \in Q^3.
  //
  // minimize   0.5 l^2 + t
  // subject to l + u = 2
  //            (t, u, v) in Q^3
  //
  // Since t >= |u| and u = 2 - l with l >= 0, the objective becomes
  // 0.5 l^2 + |2 - l|, which is minimized at l* = 1, u* = 1, t* = 1, v* = 0.
  raft::handle_t handle{};
  init_handler(&handle);

  user_problem_t<int, double> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 4;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Columns: l, t, u, v
  user_problem.A.col_start = {0, 1, 1, 2, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = 1.0;

  user_problem.rhs       = {2.0};
  user_problem.row_sense = {'E'};
  user_problem.lower.assign(n, 0.0);
  user_problem.upper.assign(n, inf);

  user_problem.Q_offsets = {0, 1, 1, 1, 1};
  user_problem.Q_indices = {0};
  user_problem.Q_values  = {1.0};

  user_problem.num_range_rows         = 0;
  user_problem.problem_name           = "qp_with_soc_block";
  user_problem.cone_var_start         = 1;
  user_problem.second_order_cone_dims = {3};
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  simplex_solver_settings_t<int, double> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<int, double> solution(m, n);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 1.5, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);
  EXPECT_NEAR(std::abs(solution.x[3]), 0.0, 1e-4);
}

}  // namespace cuopt::mathematical_optimization::simplex::test
