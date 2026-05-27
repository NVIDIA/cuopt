/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cstdio>

#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/user_problem.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>

#include <cuopt/linear_programming/io/parser.hpp>
#include <utilities/logger.hpp>

namespace cuopt::linear_programming::dual_simplex::test {

// This serves as both a warm up but also a mandatory initial call to setup cuSparse and cuBLAS
static void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

TEST(barrier, chess_set)
{
  cuopt::init_logger_t log("", true);
  namespace dual_simplex = cuopt::linear_programming::dual_simplex;
  raft::handle_t handle{};
  init_handler(&handle);
  dual_simplex::user_problem_t<int, double> user_problem(&handle);
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
  EXPECT_EQ((dual_simplex::solve_linear_program_with_barrier(user_problem, settings, solution)),
            dual_simplex::lp_status_t::OPTIMAL);
  const double objective = -solution.objective;
  EXPECT_NEAR(objective, 1333.33, 1e-2);
  EXPECT_NEAR(solution.x[0], 0.0, 1e-6);
  EXPECT_NEAR(solution.x[1], 66.6667, 1e-3);
}

TEST(barrier, dual_variable_greater_than)
{
  cuopt::init_logger_t log("", true);
  // minimize   3*x0 + 2 * x1
  // subject to  x0 + x1  >= 1
  //             x0 + 2x1 >= 3
  //             x0, x1 >= 0

  raft::handle_t handle{};
  init_handler(&handle);
  cuopt::linear_programming::dual_simplex::user_problem_t<int, double> user_problem(&handle);
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
  EXPECT_EQ((dual_simplex::solve_linear_program_with_barrier(user_problem, settings, solution)),
            dual_simplex::lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 3.0, 1e-5);
  EXPECT_NEAR(solution.x[0], 0.0, 1e-5);
  EXPECT_NEAR(solution.x[1], 1.5, 1e-5);
  EXPECT_NEAR(solution.y[0], 0.0, 1e-5);
  EXPECT_NEAR(solution.y[1], 1.0, 1e-5);
  EXPECT_NEAR(solution.z[0], 2.0, 1e-5);
  EXPECT_NEAR(solution.z[1], 0.0, 1e-5);
}

TEST(barrier, min_x_squared_free_variable_dual_correction)
{
  // minimize   x^2         (Q = [2.0], so 0.5 * x^T Q x = x^2)
  // subject to x >= 1
  // x is free
  //
  // Optimal: x = 1, obj = 1, y[0] = 2, z[0] = 0
  // This tests the dual correction for originally-free variables that
  // received implied bounds during presolve.

  const raft::handle_t handle{};
  init_handler(&handle);

  auto path =
    cuopt::test::get_rapids_dataset_root_dir() + "/quadratic_programming/min_x_squared.mps";
  auto mps_data = cuopt::linear_programming::io::read_mps<int, double>(path);

  auto settings = cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};

  auto solution = cuopt::linear_programming::solve_lp(&handle, mps_data, settings);

  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERMINATION_STATUS_OPTIMAL);

  auto h_x = cuopt::host_copy(solution.get_primal_solution(), handle.get_stream());
  auto h_y = cuopt::host_copy(solution.get_dual_solution(), handle.get_stream());
  auto h_z = cuopt::host_copy(solution.get_reduced_cost(), handle.get_stream());

  printf("x %e y %e z %e\n", h_x[0], h_y[0], h_z[0]);

  const double tol = 1e-5;
  EXPECT_NEAR(h_x[0], 1.0, tol);
  EXPECT_NEAR(h_y[0], 2.0, tol);
  EXPECT_NEAR(h_z[0], 0.0, tol);
}

}  // namespace cuopt::linear_programming::dual_simplex::test
