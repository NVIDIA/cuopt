/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <barrier/translate_soc.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/user_problem.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>

#include <cmath>
#include <iostream>
#include <vector>

namespace cuopt::linear_programming::detail::test {

using i_t  = int;
using f_t  = double;
using qc_t = optimization_problem_interface_t<i_t, f_t>::quadratic_constraint_t;
using qc_soc_path_t =
  cuopt::linear_programming::dual_simplex::user_problem_t<i_t, f_t>::qc_soc_recognition_path_t;

static void init_handler(const raft::handle_t* handle_ptr)
{
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

/** Barrier solve on expanded SOC model, then project duals/primals back to the user QCQP. */
static void solve_qc_soc_barrier_with_dual_recovery(
  cuopt::linear_programming::dual_simplex::user_problem_t<i_t, f_t>& user_problem,
  cuopt::linear_programming::dual_simplex::lp_solution_t<i_t, f_t>& solution)
{
  using namespace cuopt::linear_programming::dual_simplex;
  simplex_solver_settings_t<i_t, f_t> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  const auto status = solve_linear_program_with_barrier(user_problem, settings, solution);
  ASSERT_EQ(status, lp_status_t::OPTIMAL);

  project_barrier_qcqp_duals_to_model(user_problem, solution);
  project_barrier_solution_to_model_variables(user_problem, solution);
}

static f_t qc_dual_from_projected_solution(
  const cuopt::linear_programming::dual_simplex::user_problem_t<i_t, f_t>& user_problem,
  const cuopt::linear_programming::dual_simplex::lp_solution_t<i_t, f_t>& solution,
  i_t qc_index = 0)
{
  const i_t m_linear = user_problem.original_num_rows;
  return solution.y[m_linear + qc_index];
}

static void print_lp_solution_debug(
  const char* label,
  const cuopt::linear_programming::dual_simplex::lp_solution_t<i_t, f_t>& solution)
{
  std::cout << "=== " << label << " ===\n";
  std::cout << "  objective = " << solution.objective << '\n';
  std::cout << "  x[" << solution.x.size() << "]:";
  for (size_t i = 0; i < solution.x.size(); ++i) {
    std::cout << ' ' << i << ':' << solution.x[i];
  }
  std::cout << '\n';
  std::cout << "  y[" << solution.y.size() << "]:";
  for (size_t i = 0; i < solution.y.size(); ++i) {
    std::cout << ' ' << i << ':' << solution.y[i];
  }
  std::cout << '\n';
  std::cout << "  z[" << solution.z.size() << "]:";
  for (size_t i = 0; i < solution.z.size(); ++i) {
    std::cout << ' ' << i << ':' << solution.z[i];
  }
  std::cout << '\n';
}

// Test: general convex quadratic constraint with dense PD Q matrix.
// minimize x0 + x1
// subject to x^T Q x <= 1   where Q = [2 1; 1 2] (PD, eigenvalues 1 and 3)
// The feasible set is an ellipse centered at origin.
// Optimal should be at the boundary of the ellipse along direction (-1, -1).
// For Q = [2 1; 1 2], the minimum of (x0+x1) on x^T Q x <= 1 is:
//   c = (1,1), Q^{-1} c = (1/3)(1,1), c^T Q^{-1} c = 2/3
//   min c^T x = -sqrt(c^T Q^{-1} c) = -sqrt(2/3)
TEST(general_quadratic, dense_pd_2x2_solve)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  // Need at least one linear constraint for the barrier solver.
  // Use equality x0 - x1 = 0 to force x0 = x1.
  // With x0=x1=t: x^T[2 1;1 2]x = 6t^2 <= 1, obj = 2t.
  // Optimal: t = -1/sqrt(6), obj = -2/sqrt(6) = -sqrt(2/3).
  constexpr int m  = 1;
  constexpr int n  = 2;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 1.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = -1.0;

  user_problem.rhs            = {0.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf};
  user_problem.upper          = {inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  // Build quadratic constraint: x^T [2 1; 1 2] x <= 1
  // Q in COO (lower triangular stored):
  // (0,0,2), (1,0,1), (1,1,2)
  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "ellipse";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 1.0;
  qc.rows                 = {0, 1, 1};
  qc.cols                 = {0, 0, 1};
  qc.vals                 = {2.0, 1.0, 2.0};

  // Convert to CSR for translation (must include the linear constraint row)
  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 2};
  csr_A.j         = {0, 1};
  csr_A.x         = {1.0, -1.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);

  // Convert CSR back to CSC for the barrier solver
  csr_A.to_compressed_col(user_problem.A);

  // Verify a cone was created
  EXPECT_GT(user_problem.second_order_cone_dims.size(), 0u);
  EXPECT_GT(user_problem.cone_var_start, 0);

  // Debug: verify user_problem dimensions are consistent
  EXPECT_EQ(user_problem.A.m, user_problem.num_rows);
  EXPECT_EQ(user_problem.A.n, user_problem.num_cols);
  EXPECT_EQ(static_cast<int>(user_problem.objective.size()), user_problem.num_cols);
  EXPECT_EQ(static_cast<int>(user_problem.lower.size()), user_problem.num_cols);
  EXPECT_EQ(static_cast<int>(user_problem.upper.size()), user_problem.num_cols);
  EXPECT_EQ(static_cast<int>(user_problem.var_types.size()), user_problem.num_cols);
  EXPECT_EQ(static_cast<int>(user_problem.rhs.size()), user_problem.num_rows);
  EXPECT_EQ(static_cast<int>(user_problem.row_sense.size()), user_problem.num_rows);

  // Verify cone layout: cone vars should be a trailing block
  i_t cone_end = user_problem.cone_var_start;
  for (i_t d : user_problem.second_order_cone_dims) {
    cone_end += d;
  }
  EXPECT_EQ(cone_end, user_problem.num_cols)
    << "cone_var_start=" << user_problem.cone_var_start
    << " cone_dims sum=" << (cone_end - user_problem.cone_var_start)
    << " num_cols=" << user_problem.num_cols;

  // Debug: verify row senses are all 'E' (no inequality rows that would generate slacks)
  int n_L_rows = 0;
  for (int i = 0; i < user_problem.num_rows; ++i) {
    if (user_problem.row_sense[i] == 'L') n_L_rows++;
  }
  EXPECT_EQ(n_L_rows, 0) << "Expected all rows to be equality after conversion, but found "
                         << n_L_rows << " 'L' rows out of " << user_problem.num_rows;

  // Now solve via barrier
  simplex_solver_settings_t<i_t, f_t> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  // min (x0+x1) s.t. (1/2)*x^T*[4,1;1,4]*x <= 1 with x0=x1
  // With x0=x1=t: (1/2)*(4t^2+t^2+t^2+4t^2) = 5t^2 <= 1
  // Min 2t at t = -1/sqrt(5), obj = -2/sqrt(5)
  EXPECT_NEAR(solution.objective, -2.0 / std::sqrt(5.0), 1e-4);
}

// Test: non-convex quadratic constraint should be rejected.
// Q = [1 2; 2 1] has eigenvalues 3 and -1 (indefinite).
TEST(general_quadratic, rejects_non_convex)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 0;
  constexpr int n  = 2;
  constexpr int nz = 0;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 0, 0};

  user_problem.rhs.clear();
  user_problem.row_sense.clear();
  user_problem.lower          = {-inf, -inf};
  user_problem.upper          = {inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  // Q COO: (0,0,1), (1,0,4), (1,1,1) → H(0,0)=2, H(1,0)=4, H(1,1)=2
  // Full H = [2 4; 4 2], eigenvalues 6 and -2 → indefinite
  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "non_convex";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 1.0;
  qc.rows                 = {0, 1, 1};
  qc.cols                 = {0, 0, 1};
  qc.vals                 = {1.0, 4.0, 1.0};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0};

  std::vector<qc_t> qcs = {qc};

  // Should throw validation error for non-convex Q
  EXPECT_THROW(
    (convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem)),
    cuopt::logic_error);
}

// Test: rank-deficient PSD Q (e.g., Q = v*v^T with v = [1, 1])
// minimize x0 + x1
// subject to (x0 + x1)^2 <= 4   (i.e., |x0 + x1| <= 2)
// Q = [1 1; 1 1] has rank 1, eigenvalues 0 and 2.
// Optimal: x0 + x1 = -2, objective = -2
TEST(general_quadratic, rank_deficient_psd_solve)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 2;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = -1.0;

  user_problem.rhs            = {0.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf};
  user_problem.upper          = {inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  // Q = [1 2; 2 1] gives H = [2 2; 2 2] (rank 1), rhs = 4
  // (1/2)*x^T*[2,2;2,2]*x = (x0+x1)^2 <= 4
  // With x0=x1=t: (2t)^2 <= 4 → t >= -1, obj = 2t = -2
  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "rank1_cone";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 4.0;
  qc.rows                 = {0, 1, 1};
  qc.cols                 = {0, 0, 1};
  qc.vals                 = {1.0, 2.0, 1.0};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 2};
  csr_A.j         = {0, 1};
  csr_A.x         = {1.0, -1.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);

  // Convert CSR back to CSC for the barrier solver
  csr_A.to_compressed_col(user_problem.A);

  // The cone dimension should be rank + 2 = 1 + 2 = 3
  ASSERT_EQ(user_problem.second_order_cone_dims.size(), 1u);
  EXPECT_EQ(user_problem.second_order_cone_dims[0], 3);

  // Solve
  simplex_solver_settings_t<i_t, f_t> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  // x0=x1=t from equality. Objective = x0 = t.
  // Quadratic form from COO (0,0,1),(1,0,2),(1,1,1): H=[2,2;2,2].
  // (1/2)*x^T*H*x = (1/2)*(2t^2+2t^2+2t^2+2t^2) = 4t^2 <= 4, so t >= -1.
  // min x0 = min t = -1.
  EXPECT_NEAR(solution.objective, -1.0, 1e-4);
}

// Test: general quadratic constraint WITH an inequality linear constraint.
// minimize x0 + x1
// subject to x^T [2 1; 1 2] x <= 1   (quadratic, via general path)
//            x0 + x1 <= 10            (linear inequality)
//            x0 - x1 = 0             (linear equality)
// The inequality is inactive at optimum, so same answer as dense_pd_2x2_solve.
TEST(general_quadratic, with_inequality_constraint)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  // 2 constraints: one equality, one inequality
  constexpr int m  = 2;
  constexpr int n  = 2;
  constexpr int nz = 4;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 1.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  // Col 0: rows 0 and 1. Col 1: rows 0 and 1.
  user_problem.A.col_start = {0, 2, 4};
  user_problem.A.i[0]      = 0;  // row 0: x0 - x1 = 0
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 1;  // row 1: x0 + x1 <= 10
  user_problem.A.x[1]      = 1.0;
  user_problem.A.i[2]      = 0;  // row 0: x0 - x1 = 0
  user_problem.A.x[2]      = -1.0;
  user_problem.A.i[3]      = 1;  // row 1: x0 + x1 <= 10
  user_problem.A.x[3]      = 1.0;

  user_problem.rhs            = {0.0, 10.0};
  user_problem.row_sense      = {'E', 'L'};
  user_problem.lower          = {-inf, -inf};
  user_problem.upper          = {inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  // Q COO: (0,0,2), (1,0,1), (1,1,2) → H = [4,1;1,4]
  // (1/2) x^T [4,1;1,4] x <= 1
  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "ellipse_ineq";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 1.0;
  qc.rows                 = {0, 1, 1};
  qc.cols                 = {0, 0, 1};
  qc.vals                 = {2.0, 1.0, 2.0};

  // Build CSR matching the A matrix
  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 2, 4};
  csr_A.j         = {0, 1, 0, 1};
  csr_A.x         = {1.0, -1.0, 1.0, 1.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);

  // Convert CSR back to CSC for the barrier solver
  csr_A.to_compressed_col(user_problem.A);

  // Verify cone layout
  EXPECT_GT(user_problem.second_order_cone_dims.size(), 0u);
  i_t cone_end = user_problem.cone_var_start;
  for (i_t d : user_problem.second_order_cone_dims) {
    cone_end += d;
  }
  EXPECT_EQ(cone_end, user_problem.num_cols)
    << "Cone must be trailing block: cone_var_start=" << user_problem.cone_var_start
    << " cone_end=" << cone_end << " num_cols=" << user_problem.num_cols;

  // Solve
  simplex_solver_settings_t<i_t, f_t> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  // Same as dense_pd_2x2 test: -2/sqrt(5)
  EXPECT_NEAR(solution.objective, -2.0 / std::sqrt(5.0), 1e-4);
}

// Test: minimize t subject to ||A*x - b||^2 <= t, with b = A*e (b in range of A).
// Since b is achievable, optimal t* = 0 (at x = e).
//
// A = [1 1; 1 -1; 0 1] (3x2), e = [1, 1], b = A*e = [2, 0, 1]
// A^T*A = [2 0; 0 3], A^T*b = [2, 3], b^T*b = 5
// Variables: z = (x0, x1, t). Objective: min t → c_obj = (0, 0, 1).
// Quadratic constraint: x^T*(A^T*A)*x - 2*(A^T*b)^T*x - t <= -b^T*b
//   Q COO (3x3, only x-block): (0,0,2), (1,1,3)
//   linear: (-4, -6, -1)
//   rhs: -5
TEST(general_quadratic, least_squares_b_in_range)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  // Variables: x0, x1, u (u = t - b^T*b = t - 5).
  // Linear constraint: x0 + x1 = 2 (one row of Ax = b, helps bound x)
  constexpr int m  = 1;
  constexpr int n  = 3;  // x0, x1, u
  constexpr int nz = 2;

  user_problem.num_rows     = m;
  user_problem.num_cols     = n;
  user_problem.objective    = {0.0, 0.0, 1.0};  // minimize u (obj = u + 5 = t)
  user_problem.obj_constant = 5.0;

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = 1.0;

  user_problem.rhs            = {2.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf, -5.0};  // u >= -5 (since u = t-5, t >= 0)
  user_problem.upper          = {inf, inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  // Quadratic constraint: x^T*(A^T*A)*x - 2*(A^T*b)^T*x - u <= 0
  // This is equivalent to ||Ax - b||^2 <= t (with t = s + b^T*b, but we use t directly).
  // Reformulated with rhs = 0: x^T*Q*x + c^T*z <= 0
  // Q COO: (0,0,2), (1,1,3) — diagonal of A^T*A
  // linear: (-4, -6, -1)  — (-2*A^T*b on x, -1 on t)
  // rhs: 0
  // Note: ||Ax-b||^2 = x^T*Q*x - 2*(A^T*b)^T*x + b^T*b, so constraint is
  //   ||Ax-b||^2 - t <= 0 ⟺ x^T*Q*x - 2*(A^T*b)^T*x + b^T*b - t <= 0
  // We absorb b^T*b into the objective: min t, with t = s + b^T*b where s is our variable.
  // Simpler: just use variables (x0, x1, s) where s = t - b^T*b, minimize s (obj* = -b^T*b + t*).
  // Actually simplest: since b=A*e, optimal is t*=0, just verify obj near 0.
  // Let's use rhs = 0 formulation: x^T*Q*x - 2*(A^T*b)^T*x + 5 - t <= 0
  // i.e. x^T*Q*x - 2*(A^T*b)^T*x - t <= -5 ... that's negative rhs again.
  //
  // Alternative: use variable substitution. Let u = t - 5. Then constraint:
  //   x^T*Q*x - 4*x0 - 6*x1 - u <= 0 (rhs = 0), and objective = min(u + 5).
  // At optimum x=(1,1), constraint: 2+3-4-6-u<=0 → -5-u<=0 → u>=-5, so min u=-5, obj=0.
  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "least_squares";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 0.0;
  qc.rows                 = {0, 1};
  qc.cols                 = {0, 1};
  qc.vals                 = {2.0, 3.0};
  qc.linear_values        = {-4.0, -6.0, -1.0};
  qc.linear_indices       = {0, 1, 2};

  // Build CSR with the linear constraint: x0 + x1 = 2
  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 2};
  csr_A.j         = {0, 1};
  csr_A.x         = {1.0, 1.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);
  csr_A.to_compressed_col(user_problem.A);

  // Verify cone layout
  i_t cone_end_ls = user_problem.cone_var_start;
  for (i_t d : user_problem.second_order_cone_dims) {
    cone_end_ls += d;
  }
  EXPECT_EQ(cone_end_ls, user_problem.num_cols)
    << "cone_var_start=" << user_problem.cone_var_start << " cone_end=" << cone_end_ls
    << " num_cols=" << user_problem.num_cols;

  // Check that cone variables have valid bounds for barrier
  for (i_t j = user_problem.cone_var_start; j < user_problem.num_cols; ++j) {
    EXPECT_TRUE(user_problem.lower[j] == 0.0 || user_problem.lower[j] <= -1e30)
      << "cone var " << j << " has invalid lower=" << user_problem.lower[j];
    EXPECT_TRUE(user_problem.upper[j] >= 1e30)
      << "cone var " << j << " has invalid upper=" << user_problem.upper[j];
  }

  simplex_solver_settings_t<i_t, f_t> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  // b is in range(A) so optimal t* = 0.
  // solution.objective = u* = -5; total objective = u* + obj_constant = -5 + 5 = 0
  EXPECT_NEAR(solution.objective + user_problem.obj_constant, 0.0, 1e-3);
}

// Test: minimize t subject to ||A*x - b||^2 <= t, with b NOT in range(A).
// Optimal t* = ||A*x* - b||^2 > 0 where x* is the least-squares solution.
//
// A = [1 0; 0 1; 0 0] (3x2), b = [1, 1, 1]
// A^T*A = I_2, A^T*b = [1, 1], b^T*b = 3
// Least-squares solution: x* = A^T*b = [1, 1], residual = b - A*x* = [0, 0, 1]
// Optimal t* = ||residual||^2 = 1
//
// Variables: z = (x0, x1, t). Objective: min t.
// Q COO: (0,0,1), (1,1,1) — identity A^T*A
// linear: (-2, -2, -1) — i.e. -2*A^T*b and -1 for t
// rhs: -3 — i.e. -b^T*b
TEST(general_quadratic, least_squares_b_not_in_range)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 3;  // x0, x1, t
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 0.0, 1.0};  // minimize t

  // Equality: x0 + x1 = 2 (bounds x, but the LS solution x*=[1,1] satisfies this)
  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = 1.0;

  user_problem.rhs            = {2.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf, 0.0};  // t >= 0
  user_problem.upper          = {inf, inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  // Quadratic constraint: x^T*I*x - 2*[1,1]*x - t <= -3
  // i.e. x0^2 + x1^2 - 2*x0 - 2*x1 - t <= -3
  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "least_squares_nofit";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = -3.0;
  qc.rows                 = {0, 1};
  qc.cols                 = {0, 1};
  qc.vals                 = {1.0, 1.0};
  qc.linear_values        = {-2.0, -2.0, -1.0};
  qc.linear_indices       = {0, 1, 2};

  // Build CSR: x0 + x1 = 2
  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 2};
  csr_A.j         = {0, 1};
  csr_A.x         = {1.0, 1.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);
  csr_A.to_compressed_col(user_problem.A);

  simplex_solver_settings_t<i_t, f_t> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  auto status = solve_linear_program_with_barrier(user_problem, settings, solution);

  EXPECT_EQ(status, lp_status_t::OPTIMAL);
  // b is NOT in range(A), optimal t* = ||residual||^2 = 1
  EXPECT_NEAR(solution.objective, 1.0, 1e-3);
}

// Test: x0^2 + x1^2 - t^2 <= 0 with t >= 0 should be accepted (valid SOC: ||x|| <= t).
TEST(general_quadratic, soc_head_nonneg_accepted)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  // Variables: x0, x1, t. Constraint: x0^2 + x1^2 - t^2 <= 0
  constexpr int m  = 1;
  constexpr int n  = 3;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0, 0.0};  // minimize x0

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 0, 0, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;  // dummy row for barrier: t <= 10
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = 0.0;  // placeholder

  // Actually just use: x1 = 1 as a simple equality
  user_problem.A.col_start = {0, 0, 1, 1};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;

  user_problem.rhs            = {1.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf, 0.0};  // t >= 0
  user_problem.upper          = {inf, inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  // Q COO: x0^2 + x1^2 - t^2 <= 0
  // (0,0,1), (1,1,1), (2,2,-1)
  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "soc_valid";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 0.0;
  qc.rows                 = {0, 1, 2};
  qc.cols                 = {0, 1, 2};
  qc.vals                 = {1.0, 1.0, -1.0};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, 1);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 1};
  csr_A.j         = {1};
  csr_A.x         = {1.0};

  std::vector<qc_t> qcs = {qc};
  // Should NOT throw — head variable t has lower >= 0
  EXPECT_NO_THROW(
    (convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem)));

  // Verify it produced a cone
  EXPECT_GT(user_problem.second_order_cone_dims.size(), 0u);
}

// Test: x0^2 + x1^2 - t^2 <= 0 with t free should be rejected (non-convex without t >= 0).
TEST(general_quadratic, soc_head_free_rejected)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 3;
  constexpr int nz = 1;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 0, 1, 1};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;

  user_problem.rhs            = {1.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf, -inf};  // t is FREE — no lower bound
  user_problem.upper          = {inf, inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  // Q COO: x0^2 + x1^2 - t^2 <= 0 (same Q, but t is free)
  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "soc_invalid";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 0.0;
  qc.rows                 = {0, 1, 2};
  qc.cols                 = {0, 1, 2};
  qc.vals                 = {1.0, 1.0, -1.0};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 1};
  csr_A.j         = {1};
  csr_A.x         = {1.0};

  std::vector<qc_t> qcs = {qc};
  // Head variable t is free with no constraint implying t >= 0, so this is non-convex.
  EXPECT_THROW(
    (convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem)),
    cuopt::logic_error);
}

// Test: x0^2 + x1^2 - 2*y*z <= 0 with y >= 0, z >= 0 should be accepted (valid rotated SOC).
TEST(general_quadratic, rotated_soc_heads_nonneg_accepted)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  // Variables: x0, x1, y, z. Constraint: x0^2 + x1^2 - 2*y*z <= 0
  constexpr int m  = 1;
  constexpr int n  = 4;
  constexpr int nz = 1;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 0, 1, 1, 1};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;

  user_problem.rhs            = {1.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf, 0.0, 0.0};  // y >= 0, z >= 0
  user_problem.upper          = {inf, inf, inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  // Q COO: x0^2 + x1^2 - 2*y*z <= 0
  // Diagonal: (0,0,1), (1,1,1). Off-diagonal: (2,3,-1), (3,2,-1)
  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "rsoc_valid";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 0.0;
  qc.rows                 = {0, 1, 2, 3};
  qc.cols                 = {0, 1, 3, 2};
  qc.vals                 = {1.0, 1.0, -1.0, -1.0};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 1};
  csr_A.j         = {1};
  csr_A.x         = {1.0};

  std::vector<qc_t> qcs = {qc};
  // Should NOT throw — both head variables y and z have lower >= 0
  EXPECT_NO_THROW(
    (convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem)));

  EXPECT_GT(user_problem.second_order_cone_dims.size(), 0u);
  ASSERT_EQ(user_problem.qc_dual_recovery.size(), 1u);
  const auto& meta = user_problem.qc_dual_recovery[0];
  using path_t     = user_problem_t<i_t, f_t>::qc_soc_recognition_path_t;
  EXPECT_EQ(meta.path, path_t::ROTATED);
  EXPECT_GE(meta.rsoc_s0_lift_row, 0);
  EXPECT_GE(meta.rsoc_s1_lift_row, 0);
  EXPECT_GT(meta.rsoc_head_lift_h, 0.0);
  EXPECT_FALSE(meta.rsoc_head1_is_constant_half);
}

// Test: x0^2 + x1^2 - 2*y*z <= 0 with y free, z free should be rejected (non-convex).
TEST(general_quadratic, rotated_soc_heads_free_rejected)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 4;
  constexpr int nz = 1;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 0, 1, 1, 1};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;

  user_problem.rhs            = {1.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf, -inf, -inf};  // y and z are FREE
  user_problem.upper          = {inf, inf, inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  // Q COO: same as above
  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "rsoc_invalid";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 0.0;
  qc.rows                 = {0, 1, 2, 3};
  qc.cols                 = {0, 1, 3, 2};
  qc.vals                 = {1.0, 1.0, -1.0, -1.0};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 1};
  csr_A.j         = {1};
  csr_A.x         = {1.0};

  std::vector<qc_t> qcs = {qc};
  // Head variables y and z are free with no constraints implying non-negativity.
  EXPECT_THROW(
    (convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem)),
    cuopt::logic_error);
}

// End-to-end barrier dual recovery (mirrors python/cuopt/cuopt/tests/socp/test_socp.py path tests).
// min t  s.t. x1=1, x2=0, x1^2+x2^2 <= t^2, t>=0  =>  t=1, mu_QC=1/2.
TEST(qc_dual_recovery, lorentz_path)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 3;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 0.0, 1.0};  // min t

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 1;
  user_problem.A.x[1]      = 1.0;

  user_problem.rhs            = {1.0, 0.0};
  user_problem.row_sense      = {'E', 'E'};
  user_problem.lower          = {-inf, -inf, 0.0};
  user_problem.upper          = {inf, inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "lorentz_soc";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 0.0;
  qc.rows                 = {0, 1, 2};
  qc.cols                 = {0, 1, 2};
  qc.vals                 = {1.0, 1.0, -1.0};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 1, 2};
  csr_A.j         = {0, 1};
  csr_A.x         = {1.0, 1.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);
  csr_A.to_compressed_col(user_problem.A);

  ASSERT_EQ(user_problem.qc_dual_recovery.size(), 1u);
  EXPECT_EQ(user_problem.qc_dual_recovery[0].path, qc_soc_path_t::LORENTZ);

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  solve_qc_soc_barrier_with_dual_recovery(user_problem, solution);

  EXPECT_NEAR(solution.objective, 1.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 1.0, 1e-4);  // t
  EXPECT_NEAR(qc_dual_from_projected_solution(user_problem, solution), 0.5, 1e-4);
}

// min t  s.t. x1=0, x2=0, x1^2+x2^2 <= t^2, t>=0  =>  apex; mu_QC exported as 0.
TEST(qc_dual_recovery, lorentz_degenerate)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 3;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 0.0, 1.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 1;
  user_problem.A.x[1]      = 1.0;

  user_problem.rhs            = {0.0, 0.0};
  user_problem.row_sense      = {'E', 'E'};
  user_problem.lower          = {-inf, -inf, 0.0};
  user_problem.upper          = {inf, inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "lorentz_soc";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 0.0;
  qc.rows                 = {0, 1, 2};
  qc.cols                 = {0, 1, 2};
  qc.vals                 = {1.0, 1.0, -1.0};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 1, 2};
  csr_A.j         = {0, 1};
  csr_A.x         = {1.0, 1.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);
  csr_A.to_compressed_col(user_problem.A);

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  solve_qc_soc_barrier_with_dual_recovery(user_problem, solution);

  EXPECT_NEAR(solution.objective, 0.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 0.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 0.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 0.0, 1e-4);
  EXPECT_NEAR(qc_dual_from_projected_solution(user_problem, solution), 0.0, 1e-4);
}

// min x1  s.t. x0=1, x1^2+x2^2 <= x0  =>  x1=-1, mu_QC=1/2.
TEST(qc_dual_recovery, affine_path)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 3;  // x0, x1, x2
  constexpr int nz = 1;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {0.0, 1.0, 0.0};  // min x1

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 1, 1};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;

  user_problem.rhs            = {1.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf, -inf};
  user_problem.upper          = {inf, inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "affine_soc";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 0.0;
  qc.rows                 = {1, 2};  // x1^2 + x2^2
  qc.cols                 = {1, 2};
  qc.vals                 = {1.0, 1.0};
  qc.linear_indices       = {0};  // -x0
  qc.linear_values        = {-1.0};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 1};
  csr_A.j         = {0};
  csr_A.x         = {1.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);
  csr_A.to_compressed_col(user_problem.A);

  ASSERT_EQ(user_problem.qc_dual_recovery.size(), 1u);
  EXPECT_EQ(user_problem.qc_dual_recovery[0].path, qc_soc_path_t::AFFINE);
  EXPECT_GE(user_problem.qc_dual_recovery[0].affine_link_row, 0);

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  solve_qc_soc_barrier_with_dual_recovery(user_problem, solution);

  EXPECT_NEAR(solution.objective, -1.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 1.0, 1e-4);   // x0
  EXPECT_NEAR(solution.x[1], -1.0, 1e-4);  // x1
  EXPECT_NEAR(solution.x[2], 0.0, 1e-4);   // x2
  EXPECT_NEAR(solution.y[0], -0.5, 1e-4);  // EQ x0 = 1
  EXPECT_NEAR(qc_dual_from_projected_solution(user_problem, solution), 0.5, 1e-4);
}

// min x0+x1  s.t. x0=0, x1=0, x2^2+x3^2 <= x0*x1, x0,x1>=0  =>  lifted s0=s1=0; mu_QC=0.
TEST(qc_dual_recovery, rotated_degenerate)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 4;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 1.0, 0.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2, 2, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 1;
  user_problem.A.x[1]      = 1.0;

  user_problem.rhs            = {0.0, 0.0};
  user_problem.row_sense      = {'E', 'E'};
  user_problem.lower          = {0.0, 0.0, -inf, -inf};
  user_problem.upper          = {inf, inf, inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "rotated_soc";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 0.0;
  qc.rows                 = {2, 3, 0, 1};
  qc.cols                 = {2, 3, 1, 0};
  qc.vals                 = {1.0, 1.0, -0.5, -0.5};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 1, 2};
  csr_A.j         = {0, 1};
  csr_A.x         = {1.0, 1.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);
  csr_A.to_compressed_col(user_problem.A);

  ASSERT_EQ(user_problem.qc_dual_recovery.size(), 1u);
  EXPECT_EQ(user_problem.qc_dual_recovery[0].path, qc_soc_path_t::ROTATED);

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  solve_qc_soc_barrier_with_dual_recovery(user_problem, solution);

  EXPECT_NEAR(solution.objective, 0.0, 1e-4);
  EXPECT_NEAR(solution.x[0], 0.0, 1e-4);
  EXPECT_NEAR(solution.x[1], 0.0, 1e-4);
  EXPECT_NEAR(solution.x[2], 0.0, 1e-4);
  EXPECT_NEAR(solution.x[3], 0.0, 1e-4);
  EXPECT_NEAR(qc_dual_from_projected_solution(user_problem, solution), 0.0, 1e-4);
}

// min x0+3*x2  s.t. x1=2, x3=0, x2^2+x3^2 <= x0*x1, x0,x1>=0  =>  obj=-9/2, mu_QC=1/2.
TEST(qc_dual_recovery, rotated_path)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 2;
  constexpr int n  = 4;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 0.0, 3.0, 0.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 0, 1, 1, 1};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 1;
  user_problem.A.x[1]      = 1.0;

  user_problem.rhs            = {2.0, 0.0};
  user_problem.row_sense      = {'E', 'E'};
  user_problem.lower          = {0.0, 0.0, -inf, -inf};
  user_problem.upper          = {inf, inf, inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "rotated_soc";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 0.0;
  qc.rows                 = {2, 3, 0, 1};
  qc.cols                 = {2, 3, 1, 0};
  qc.vals                 = {1.0, 1.0, -0.5, -0.5};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 1, 2};
  csr_A.j         = {1, 3};
  csr_A.x         = {1.0, 1.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);
  csr_A.to_compressed_col(user_problem.A);

  ASSERT_EQ(user_problem.qc_dual_recovery.size(), 1u);
  EXPECT_EQ(user_problem.qc_dual_recovery[0].path, qc_soc_path_t::ROTATED);

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);
  solve_qc_soc_barrier_with_dual_recovery(user_problem, solution);

  EXPECT_NEAR(solution.objective, -4.5, 1e-4);
  EXPECT_NEAR(solution.x[0], 4.5, 1e-4);
  EXPECT_NEAR(solution.x[1], 2.0, 1e-4);
  EXPECT_NEAR(solution.x[2], -3.0, 1e-4);
  EXPECT_NEAR(solution.x[3], 0.0, 1e-4);
  EXPECT_NEAR(qc_dual_from_projected_solution(user_problem, solution), 0.5, 1e-4);
}

// min 2*x0+x1  s.t. 2*x0^2+2*x0*x1+2*x1^2<=1, x0-3*x1=0
//   =>  x0=-3/sqrt(26), x1=-1/sqrt(26), mu_QC=7/(2*sqrt(26)), y_KKT=-3/26.
TEST(qc_dual_recovery, general_path)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 2;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {2.0, 1.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = -3.0;

  user_problem.rhs            = {0.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf};
  user_problem.upper          = {inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "general_qc";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 1.0;
  qc.rows                 = {0, 1, 1};
  qc.cols                 = {0, 0, 1};
  qc.vals                 = {2.0, 2.0, 2.0};  // 2*x0^2 + 2*x0*x1 + 2*x1^2

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 2};
  csr_A.j         = {0, 1};
  csr_A.x         = {1.0, -3.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);
  csr_A.to_compressed_col(user_problem.A);

  ASSERT_EQ(user_problem.qc_dual_recovery.size(), 1u);
  const auto& meta = user_problem.qc_dual_recovery[0];
  EXPECT_EQ(meta.path, qc_soc_path_t::GENERAL);
  EXPECT_GE(meta.s0_link_row, 0);
  EXPECT_GE(meta.sr1_link_row, 0);

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);

  simplex_solver_settings_t<i_t, f_t> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  ASSERT_EQ(solve_linear_program_with_barrier(user_problem, settings, solution),
            lp_status_t::OPTIMAL);

  std::cout << "qc_dual_recovery.general_path debug:\n";
  std::cout << "  expanded num_rows=" << user_problem.num_rows
            << " num_cols=" << user_problem.num_cols
            << " original_num_rows=" << user_problem.original_num_rows
            << " original_num_cols=" << user_problem.original_num_cols
            << " cone_var_start=" << user_problem.cone_var_start << '\n';
  std::cout << "  s0_link_row=" << meta.s0_link_row << " sr1_link_row=" << meta.sr1_link_row
            << " cone_index=" << meta.cone_index << '\n';
  if (!user_problem.second_order_cone_dims.empty()) {
    std::cout << "  cone_dim=" << user_problem.second_order_cone_dims[0] << '\n';
  }
  print_lp_solution_debug("expanded barrier (pre-projection)", solution);

  project_barrier_qcqp_duals_to_model(user_problem, solution);
  print_lp_solution_debug("after project_barrier_qcqp_duals_to_model", solution);

  project_barrier_solution_to_model_variables(user_problem, solution);
  print_lp_solution_debug("user model (final)", solution);

  const f_t inv_sqrt26 = 1.0 / std::sqrt(26.0);
  EXPECT_NEAR(solution.objective, -7.0 * inv_sqrt26, 1e-4);
  EXPECT_NEAR(solution.x[0], -3.0 * inv_sqrt26, 1e-4);
  EXPECT_NEAR(solution.x[1], -inv_sqrt26, 1e-4);
  EXPECT_GT(std::abs(solution.x[0] - solution.x[1]), 1e-3);
  EXPECT_NEAR(solution.y[0], 3.0 / 26.0, 1e-4);  // EQ x0 - 3*x1 = 0  (y_KKT = -3/26)
  EXPECT_NEAR(
    qc_dual_from_projected_solution(user_problem, solution), 7.0 * inv_sqrt26 / 2.0, 1e-4);
}

// min x0+2*x1  s.t. (x0+x1)^2<=1, x0-2*x1=0  (rank-1 PSD QC, LDLT rank r=1)
//   =>  x0=-2/3, x1=-1/3, obj=-4/3, mu_QC=2/3, y_KKT=1/3.
TEST(qc_dual_recovery, general_path_rank_one)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 2;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 2.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = -2.0;

  user_problem.rhs            = {0.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf};
  user_problem.upper          = {inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  // (x0+x1)^2 <= 1  via Q COO (0,0,1), (1,0,2), (1,1,1)  ->  H = [2,2;2,2], rank 1.
  // Cross term needs coefficient 2 (not 1): vals {1,1,1} would give x0^2+x0*x1+x1^2, full rank.
  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "rank1_general_qc";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 1.0;
  qc.rows                 = {0, 1, 1};
  qc.cols                 = {0, 0, 1};
  qc.vals                 = {1.0, 2.0, 1.0};

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 2};
  csr_A.j         = {0, 1};
  csr_A.x         = {1.0, -2.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);
  csr_A.to_compressed_col(user_problem.A);

  ASSERT_EQ(user_problem.qc_dual_recovery.size(), 1u);
  const auto& meta = user_problem.qc_dual_recovery[0];
  EXPECT_EQ(meta.path, qc_soc_path_t::GENERAL);
  EXPECT_GE(meta.s0_link_row, 0);
  EXPECT_GE(meta.sr1_link_row, 0);
  ASSERT_EQ(user_problem.second_order_cone_dims.size(), 1u);
  EXPECT_EQ(user_problem.second_order_cone_dims[0], 3);  // rank 1 + 2 slacks

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);

  simplex_solver_settings_t<i_t, f_t> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  ASSERT_EQ(solve_linear_program_with_barrier(user_problem, settings, solution),
            lp_status_t::OPTIMAL);

  std::cout << "qc_dual_recovery.general_path_rank_one debug:\n";
  std::cout << "  expanded num_rows=" << user_problem.num_rows
            << " num_cols=" << user_problem.num_cols
            << " original_num_rows=" << user_problem.original_num_rows
            << " original_num_cols=" << user_problem.original_num_cols
            << " cone_var_start=" << user_problem.cone_var_start << '\n';
  std::cout << "  s0_link_row=" << meta.s0_link_row << " sr1_link_row=" << meta.sr1_link_row
            << " cone_index=" << meta.cone_index << '\n';
  if (!user_problem.second_order_cone_dims.empty()) {
    std::cout << "  cone_dim=" << user_problem.second_order_cone_dims[0] << '\n';
  }
  print_lp_solution_debug("expanded barrier (pre-projection)", solution);

  project_barrier_qcqp_duals_to_model(user_problem, solution);
  print_lp_solution_debug("after project_barrier_qcqp_duals_to_model", solution);

  project_barrier_solution_to_model_variables(user_problem, solution);
  print_lp_solution_debug("user model (final)", solution);

  EXPECT_NEAR(solution.objective, -4.0 / 3.0, 1e-4);
  EXPECT_NEAR(solution.x[0], -2.0 / 3.0, 1e-4);
  EXPECT_NEAR(solution.x[1], -1.0 / 3.0, 1e-4);
  EXPECT_NEAR(solution.y[0], -1.0 / 3.0, 1e-4);  // EQ x0 - 2*x1 = 0  (y_KKT = 1/3)
  EXPECT_NEAR(qc_dual_from_projected_solution(user_problem, solution), 2.0 / 3.0, 1e-4);
}

// min x0+x1  s.t. 2*x0^2+2*x0*x1+2*x1^2+x0+x1<=3, x0-x1=0  (QC linear part c!=0)
//   =>  x0=x1=(-1-sqrt(19))/6, obj=(-1-sqrt(19))/3, mu_QC=1/sqrt(19), y_KKT=0.
TEST(qc_dual_recovery, general_path_linear_qc)
{
  raft::handle_t handle{};
  init_handler(&handle);

  using namespace cuopt::linear_programming::dual_simplex;
  user_problem_t<i_t, f_t> user_problem(&handle);

  constexpr int m  = 1;
  constexpr int n  = 2;
  constexpr int nz = 2;

  user_problem.num_rows  = m;
  user_problem.num_cols  = n;
  user_problem.objective = {1.0, 1.0};

  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start = {0, 1, 2};
  user_problem.A.i[0]      = 0;
  user_problem.A.x[0]      = 1.0;
  user_problem.A.i[1]      = 0;
  user_problem.A.x[1]      = -1.0;

  user_problem.rhs            = {0.0};
  user_problem.row_sense      = {'E'};
  user_problem.lower          = {-inf, -inf};
  user_problem.upper          = {inf, inf};
  user_problem.num_range_rows = 0;
  user_problem.var_types.assign(n, variable_type_t::CONTINUOUS);

  qc_t qc;
  qc.constraint_row_index = 0;
  qc.constraint_row_name  = "linear_general_qc";
  qc.constraint_row_type  = 'L';
  qc.rhs_value            = 3.0;
  qc.rows                 = {0, 1, 1};
  qc.cols                 = {0, 0, 1};
  qc.vals                 = {2.0, 2.0, 2.0};
  qc.linear_indices       = {0, 1};
  qc.linear_values        = {1.0, 1.0};  // + x0 + x1

  dual_simplex::csr_matrix_t<i_t, f_t> csr_A(m, n, nz);
  csr_A.m         = m;
  csr_A.n         = n;
  csr_A.row_start = {0, 2};
  csr_A.j         = {0, 1};
  csr_A.x         = {1.0, -1.0};

  std::vector<qc_t> qcs = {qc};
  convert_quadratic_constraints_to_second_order_cones<i_t, f_t>(n, qcs, csr_A, user_problem);
  csr_A.to_compressed_col(user_problem.A);

  ASSERT_EQ(user_problem.qc_dual_recovery.size(), 1u);
  const auto& meta = user_problem.qc_dual_recovery[0];
  EXPECT_EQ(meta.path, qc_soc_path_t::GENERAL);
  EXPECT_GE(meta.s0_link_row, 0);
  EXPECT_GE(meta.sr1_link_row, 0);

  lp_solution_t<i_t, f_t> solution(user_problem.num_rows, user_problem.num_cols);

  simplex_solver_settings_t<i_t, f_t> settings;
  settings.barrier          = true;
  settings.barrier_presolve = true;
  settings.dualize          = 0;
  ASSERT_EQ(solve_linear_program_with_barrier(user_problem, settings, solution),
            lp_status_t::OPTIMAL);

  std::cout << "qc_dual_recovery.general_path_linear_qc debug:\n";
  std::cout << "  expanded num_rows=" << user_problem.num_rows
            << " num_cols=" << user_problem.num_cols
            << " original_num_rows=" << user_problem.original_num_rows
            << " original_num_cols=" << user_problem.original_num_cols
            << " cone_var_start=" << user_problem.cone_var_start << '\n';
  std::cout << "  s0_link_row=" << meta.s0_link_row << " sr1_link_row=" << meta.sr1_link_row
            << " cone_index=" << meta.cone_index << '\n';
  if (!user_problem.second_order_cone_dims.empty()) {
    std::cout << "  cone_dim=" << user_problem.second_order_cone_dims[0] << '\n';
  }
  print_lp_solution_debug("expanded barrier (pre-projection)", solution);

  project_barrier_qcqp_duals_to_model(user_problem, solution);
  print_lp_solution_debug("after project_barrier_qcqp_duals_to_model", solution);

  project_barrier_solution_to_model_variables(user_problem, solution);
  print_lp_solution_debug("user model (final)", solution);

  const f_t sqrt19 = std::sqrt(19.0);
  const f_t t_star = (-1.0 - sqrt19) / 6.0;
  const f_t mu_qc  = 1.0 / sqrt19;

  EXPECT_NEAR(solution.objective, 2.0 * t_star, 1e-4);
  EXPECT_NEAR(solution.x[0], t_star, 1e-4);
  EXPECT_NEAR(solution.x[1], t_star, 1e-4);
  EXPECT_NEAR(solution.y[0], 0.0, 1e-4);  // EQ x0 - x1 = 0
  EXPECT_NEAR(qc_dual_from_projected_solution(user_problem, solution), mu_qc, 1e-4);
}

}  // namespace cuopt::linear_programming::detail::test
