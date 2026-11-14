/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/farkas_certificate.hpp>
#include <linear_algebra/sparse_matrix.hpp>
#include <linear_algebra/vector_math.hpp>
#include <math_optimization/types.hpp>

#include <algorithm>

namespace cuopt::mathematical_optimization::simplex {

// Computes a Farkas certificate of infeasibility for the primal constraints
// A*x = b, -inf <= l <= x <= u <= inf,  assuming consistent bounds (l <= u).
//
// The Farkas certificate is a triple (y, zl, zu) such that
//
// A^T*y + zl - zu = 0,   zl >= 0, zu >= 0, lj = -inf => zl_j = 0, uj = inf => zu_j = 0
// C = b^T*y + sum_{j: l_j > -inf} l_j * zl_j - sum_{j: u_j < inf} u_j * zu_j > 0
//
// The above is in exact arithmetic, in practice we need to handle that
//
// A^T y + zl - zu = r != 0
//
// Suppose a feasible x exists, then l <= x <= u.
// Since zl >= 0, we have that sum_{j: l_j > -inf} x_j * zl_j >= sum_{j: l_j > -inf} l_j * zl_j.
// Simiarly, since zu >= 0, we have that sum_{j: u_j < inf} x_j * zu_j <= sum_{j: u_j < inf} u_j *
// zu_j or                                   -sum_{j: u_j < inf} x_j * zu_j >= -sum_{j: u_j < inf}
// u_j * zu_j
//
// Therefore, C = b^T * y + sum_{j: l_j > -inf} l_j * zl_j - sum_{j: u_j < inf} u_j * zu_j
//             <= b^T * y + sum_{j: l_j > -inf} x_j * zl_j - sum_{j: u_j < inf} u_j * zu_j
//             <= b^T * y + sum_{j: l_j > -inf} x_j * zl_j - sum_{j: u_j < inf} x_j * zu_j
// Using the fact that A*x = b or b^T * y = x^T * A^T *y, and A^T * y = r - zl + zu  we have
// C <= x^T * (r - zl + zu) + sum_{j: l_j > -inf} x_j * zl_j - sum_{j: u_j < inf} x_j * zu_j
//   <= r^T*x - sum_{j: l_j > -inf} x_j * zl_j + sum_{j: u_j < inf} x_j * zu_j
//            + sum_{j: l_j > -inf} x_j * zl_j - sum_{j: u_j < inf} x_j * zu_j
//   <= r^T*x
// Thus, ever feasible x must statisfy C <= r^T * x
//
// Define U = max r^T x subject to l <= x <= u
//
// Every feasible x must satisfy C <= r^T * x <= U
//
// If we establish that C > U, no feasible x can exist.
//
// We can compute U as U = sum_j U_j where
//
// U_j = { u_j r_j  if  r_j > 0,
//       { l_j r_j  if  r_j < 0,
//       { 0        if  r_j == 0
//
// Note that if r_j > 0 and u_j is inf, U_j = inf and we can have no certificate.
// Similarly, if r_j < 0 and l_j is -inf, U_j = inf and we can have no certificate.
template <typename i_t, typename f_t>
bool compute_farkas_certificate(const lp_problem_t<i_t, f_t>& lp,
                                const simplex_solver_settings_t<i_t, f_t>& settings,
                                const std::vector<f_t>& delta_y,
                                const std::vector<f_t>& delta_z,
                                std::vector<f_t>& farkas_y,
                                std::vector<f_t>& farkas_zl,
                                std::vector<f_t>& farkas_zu,
                                f_t& farkas_constant,
                                f_t& work_estimate)
{
  const i_t m        = lp.num_rows;
  const i_t n        = lp.num_cols;
  const f_t epsilon  = settings.primal_tol;
  const f_t zero_tol = settings.zero_tol;

  farkas_constant = 0.0;

  // Construct farkas_y, farkas_zl, farkas_zu.
  farkas_y = delta_y;
  work_estimate += m;

  farkas_zl.resize(n);
  farkas_zu.resize(n);
  for (i_t j = 0; j < n; j++) {
    farkas_zl[j] = std::max<f_t>(delta_z[j], 0.0);
    farkas_zu[j] = std::max<f_t>(-delta_z[j], 0.0);
  }
  work_estimate += 3.0 * n;

  // Compute r = A^T*farkas_y + farkas_zl - farkas_zu.
  std::vector<f_t> residual(n);
  for (i_t j = 0; j < n; j++) {
    residual[j] = farkas_zl[j] - farkas_zu[j];
  }
  work_estimate += 3.0 * n;

  matrix_transpose_vector_multiply(lp.A, 1.0, farkas_y, 1.0, residual);
  work_estimate += 2.0 * lp.A.nnz() + 2.0 * n;

  // Compute a residual bound U = max r^T*x subject to l <= x <= u.
  // Positive residuals select upper bounds; negative residuals select
  // lower bounds. Residuals within zero_tol are treated as zero.
  f_t residual_bound = 0.0;
  i_t nnz_residual   = 0;
  for (i_t j = 0; j < n; j++) {
    if (residual[j] > zero_tol) {
      if (lp.upper[j] == inf) { return false; }
      residual_bound += residual[j] * lp.upper[j];
      nnz_residual++;
    } else if (residual[j] < -zero_tol) {
      if (lp.lower[j] == -inf) { return false; }
      residual_bound += residual[j] * lp.lower[j];
      nnz_residual++;
    }
  }
  work_estimate += n + nnz_residual;

  // Compute the farkas constant C = b^T*farkas_y
  //                                + l^T*farkas_zl - u^T*farkas_zu.
  // Bound multipliers above zero_tol require finite bounds.
  farkas_constant = dot<i_t, f_t>(lp.rhs, farkas_y);
  work_estimate += 2.0 * m;

  i_t nnz_bound_multipliers = 0;
  for (i_t j = 0; j < n; j++) {
    if (farkas_zl[j] > zero_tol) {
      if (lp.lower[j] == -inf) { return false; }
      farkas_constant += lp.lower[j] * farkas_zl[j];
      nnz_bound_multipliers++;
    }

    if (farkas_zu[j] > zero_tol) {
      if (lp.upper[j] == inf) { return false; }
      farkas_constant -= lp.upper[j] * farkas_zu[j];
      nnz_bound_multipliers++;
    }
  }
  work_estimate += 2.0 * n + nnz_bound_multipliers;

  // Every feasible x must satisfy C <= r^T*x <= U.
  // Require C - U >= epsilon as a numerical margin.
  return farkas_constant - residual_bound >= epsilon;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template bool compute_farkas_certificate<int, double>(
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  const std::vector<double>& delta_y,
  const std::vector<double>& delta_z,
  std::vector<double>& farkas_y,
  std::vector<double>& farkas_zl,
  std::vector<double>& farkas_zu,
  double& farkas_constant,
  double& work_estimate);

#endif

}  // namespace cuopt::mathematical_optimization::simplex
