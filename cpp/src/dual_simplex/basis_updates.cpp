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

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/triangle_solve.hpp>

#include <cmath>
#include <limits>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::b_solve(const std::vector<f_t>& rhs, std::vector<f_t>& solution) const
{
  std::vector<f_t> Lsol;
  return b_solve(rhs, solution, Lsol);
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::b_solve(const sparse_vector_t<i_t, f_t>& rhs,
                                      sparse_vector_t<i_t, f_t>& solution) const
{
  sparse_vector_t<i_t, f_t> Lsol(rhs.n, 0);
  return b_solve(rhs, solution, Lsol);
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::b_solve(const std::vector<f_t>& rhs,
                                      std::vector<f_t>& solution,
                                      std::vector<f_t>& Lsol) const
{
  const i_t m = L0_.m;
  assert(row_permutation_.size() == m);
  assert(rhs.size() == m);
  assert(solution.size() == m);

  // P*B = L*U
  // B*x = b
  // P*B*x = P*b = b'
  permute_vector(row_permutation_, rhs, solution);

  // L*U*x = b'
  // Solve for v such that L*v = b'
  l_solve(solution);
  Lsol = solution;

  // Solve for x such that U*x = v
  u_solve(solution);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::b_solve(const sparse_vector_t<i_t, f_t>& rhs,
                                      sparse_vector_t<i_t, f_t>& solution,
                                      sparse_vector_t<i_t, f_t>& Lsol) const
{
  const i_t m = L0_.m;
  assert(row_permutation_.size() == m);
  assert(rhs.n == m);
  assert(solution.n == m);
  assert(Lsol.n == m);

  // P*B = L*U
  // B*x = b
  // P*B*x = P*b = b'
  solution = rhs;
  solution.inverse_permute_vector(inverse_row_permutation_);

  // L*U*x = b'
  // Solve for v such that L*v = b'
  l_solve(solution);
  Lsol = solution;

  // Solve for x such that U*x = v
  u_solve(solution);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::b_transpose_solve(const std::vector<f_t>& rhs,
                                                std::vector<f_t>& solution) const
{
  // Observe that
  // P*B = L*U
  // B'*P' = U'*L'
  // We want to solve
  // B'*y = c
  // Let y = P'*w
  // B'*y = B'*P'*w = U'*L'*w = c
  // 1. Solve U'*r = c for r
  // 2. Solve L'*w = r for w
  // 3. Compute y = P'*w

  const i_t m = L0_.m;
  assert(rhs.size() == m);
  assert(row_permutation_.size() == m);
  assert(solution.size() == m);

  // Solve for r such that U'*r = c
  std::vector<f_t> r = rhs;
  u_transpose_solve(r);

  // Solve for w such that L'*w = r
  l_transpose_solve(r);

  // y = P'*w
  inverse_permute_vector(row_permutation_, r, solution);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::b_transpose_solve(const sparse_vector_t<i_t, f_t>& rhs,
                                                sparse_vector_t<i_t, f_t>& solution) const
{
  // Observe that
  // P*B = L*U
  // B'*P' = U'*L'
  // We want to solve
  // B'*y = c
  // Let y = P'*w
  // B'*y = B'*P'*w = U'*L'*w = c
  // 1. Solve U'*r = c for r
  // 2. Solve L'*w = r for w
  // 3. Compute y = P'*w

  const i_t m = L0_.m;
  assert(rhs.n == m);
  assert(solution.n == m);

  // Solve for r such that U'*r = c
  // Actually Q*U'*Q'*r = c
  sparse_vector_t<i_t, f_t> r = rhs;
  u_transpose_solve(r);

#ifdef CHECK_U_TRANSPOSE_SOLVE
  std::vector<f_t> residual;
  rhs.to_dense(residual);
  std::vector<f_t> r_dense;
  r.to_dense(r_dense);
  std::vector<f_t> product(m);
  // Q * U' * Q' * r_dense - c

  std::vector<f_t> r_dense_permuted(m);
  inverse_permute_vector(col_permutation_, r_dense, r_dense_permuted);

  // product = U' * Q' * r_dense
  matrix_transpose_vector_multiply(U_, 1.0, r_dense_permuted, 0.0, product);
  std::vector<f_t> product_permuted(m);
  permute_vector(col_permutation_, product, product_permuted);
  // residual = product_permuted - c
  for (i_t k = 0; k < m; ++k) {
    residual[k] -= product_permuted[k];
  }

  const f_t Ut_error = vector_norm_inf<i_t, f_t>(residual);
  if (Ut_error > 1e-6) {
    printf("|| U' * r - c || %e\n", Ut_error);
    for (i_t k = 0; k < m; ++k) {
      if (std::abs(residual[k]) > 1e-6) { printf("%d residual %e\n", k, residual[k]); }
    }
    printf("rhs nz %d\n", rhs.i.size());
  }
#endif

  // Solve for w such that L'*w = r
  l_transpose_solve(r);

  // y = P'*w
  r.inverse_permute_vector(row_permutation_, solution);

#ifdef CHECK_PERMUTATION
  std::vector<f_t> r_dense2;
  r.to_dense(r_dense2);
  std::vector<f_t> solution_dense_permuted(m);
  permute_vector(inverse_row_permutation_, r_dense2, solution_dense_permuted);
  std::vector<f_t> solution_dense;
  solution.to_dense(solution_dense);
  bool found_error = false;
  for (i_t k = 0; k < m; ++k) {
    if (std::abs(solution_dense[k] - solution_dense_permuted[k]) > 1e-6) {
      printf("B transpose inverse permutation error %d %e %e\n",
             k,
             solution_dense[k],
             solution_dense_permuted[k]);
      found_error = true;
    }
  }
  if (found_error) {
    for (i_t k = 0; k < m; ++k) {
      printf("%d (sparse -> permuted -> dense) %e (sparse -> dense -> permuted)%e\n",
             k,
             solution_dense[k],
             solution_dense_permuted[k]);
    }
    for (i_t k = 0; k < solution.i.size(); ++k) {
      printf("%d solution sparse %d %e\n", k, solution.i[k], solution.x[k]);
    }
    for (i_t k = 0; k < m; ++k) {
      if (solution_dense[k] != 0.0) { printf("%d solution dense %e\n", k, solution_dense[k]); }
    }
    for (i_t k = 0; k < m; ++k) {
      printf("inv permutation %d %d\n", k, inverse_row_permutation_[k]);
    }
    for (i_t k = 0; k < m; ++k) {
      if (r_dense2[k] != 0.0) { printf("%d r dense %e\n", k, r_dense2[k]); }
    }
    for (i_t k = 0; k < m; ++k) {
      if (solution_dense_permuted[k] != 0.0) {
        printf("%d solution dense permuted %e\n", k, solution_dense_permuted[k]);
      }
    }
  }
#endif
  return 0;
}

// Solve for x such that L*x = b
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::l_solve(std::vector<f_t>& rhs) const
{
  // L = L0 * R1^{-1} * R2^{-1} * ... * Rk^{-1}
  //
  // where Ri = I + e_r d^T

#ifdef CHECK_LOWER_SOLVE
  std::vector<f_t> b        = rhs;
  std::vector<f_t> residual = rhs;
#endif
  // First solve
  // L0*x0 = b
  dual_simplex::lower_triangular_solve(L0_, rhs);
#ifdef CHECK_LOWER_SOLVE
  {
    matrix_vector_multiply(L0_, 1.0, rhs, -1.0, residual);
    printf("|| L0 * x - b || %e\n", vector_norm_inf(residual));
  }
#endif

  // then solve R1^{-1}*x1 = x0     ->  x1 = R1*x0
  // then solve R2^{-1}*x2 = x1     ->  x2 = R2*x1
  // until we get to
  // Rk^{-1}*x = xk-1               -> x = Rk*xk-1
  // Rk = (I + e_rk dk^T)
  // x = Rk*xk-1 = xk-1 + erk (dk^T xk-1)
  for (i_t k = 0; k < num_updates_; ++k) {
    const i_t r         = pivot_indices_[k];
    f_t dot             = 0.0;
    const i_t col_start = S_.col_start[k];
    const i_t col_end   = S_.col_start[k + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      dot += S_.x[p] * rhs[S_.i[p]];
    }
    rhs[r] += dot;
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::l_solve(sparse_vector_t<i_t, f_t>& rhs) const
{
  // L = L0 * R1^{-1} * R2^{-1} * ... * Rk^{-1}
  //
  // where Ri = I + e_r d^T

  // First solve
  // L0*x0 = b
  const i_t m = L0_.m;

  i_t top = sparse_triangle_solve<i_t, f_t, true>(
    rhs, std::nullopt, xi_workspace_, L0_, x_workspace_.data());
  solve_to_sparse_vector(top, rhs);  // Uses xi_workspace_ and x_workspace_ to fill rhs

#ifdef CHECK_L_SOLVE
  std::vector<f_t> residual(m, 0.0);
  const i_t col_start = B.col_start[0];
  const i_t col_end   = B.col_start[1];
  for (i_t p = col_start; p < col_end; ++p) {
    residual[B.i[p]] = B.x[p];
  }

  std::vector<f_t> x0;
  rhs.to_dense(x0);
  matrix_vector_multiply(L0_, 1.0, x0, -1.0, residual);
  const f_t L0_solve_error = vector_norm_inf<i_t, f_t>(residual);
  if (L0_solve_error > 1e-10) { printf("L0 solve error %e\n", L0_solve_error); }
#endif

  // then solve R1^{-1}*x1 = x0     ->  x1 = R1*x0
  // then solve R2^{-1}*x2 = x1     ->  x2 = R2*x1
  // until we get to
  // Rk^{-1}*x = xk-1               -> x = Rk*xk-1
  // Rk = (I + e_rk dk^T)
  // x = Rk*xk-1 = xk-1 + erk (dk^T xk-1)

#ifdef CHECK_MULTIPLY
  std::vector<f_t> multiply;
  rhs.to_dense(multiply);
#endif

  i_t nz = scatter_into_workspace(rhs);

  for (i_t k = 0; k < num_updates_; ++k) {
    const i_t r         = pivot_indices_[k];
    f_t dot             = 0.0;
    const i_t col_start = S_.col_start[k];
    const i_t col_end   = S_.col_start[k + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      if (xi_workspace_[S_.i[p]]) { dot += S_.x[p] * x_workspace_[S_.i[p]]; }
    }
    if (!xi_workspace_[r]) {
      xi_workspace_[r]      = 1;
      xi_workspace_[m + nz] = r;
      nz++;
    }
    x_workspace_[r] += dot;

#ifdef CHECK_MULTIPLY
    f_t dot2 = 0.0;
    for (i_t p = col_start; p < col_end; ++p) {
      dot2 += S_.x[p] * multiply[S_.i[p]];
    }
    multiply[r] += dot2;
#endif
  }

  // Gather the solution into rhs
  gather_into_sparse_vector(nz, rhs);

  rhs.sort();

#ifdef CHECK_MULTIPLY
  std::vector<f_t> rhs_dense;
  rhs.to_dense(rhs_dense);
  for (i_t k = 0; k < m; ++k) {
    if (std::abs(rhs_dense[k] - multiply[k]) > 1e-10) {
      printf("l_solve rhs dense/multiply error %d %e %e\n", k, rhs_dense[k], multiply[k]);
    }
  }
#endif

  return 0;
}

// Solve for y such that L'*y = c
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::l_transpose_solve(std::vector<f_t>& rhs) const
{
  // L = L0*R1^{-1}* ... * Rk^{-1}
  // L' = Rk^{-T} * Rk-1^{-T} * ... * R2^{-T} * R1^{-T} * L0^T
  // L'*y = c
  // Rk^{-T}* Rk-1^{-T} * ... * R2^{-T} * R1^{-T} * L0^T * y = c
  const i_t m = L0_.m;
  for (i_t k = num_updates_ - 1; k >= 0; --k) {
    const i_t r = pivot_indices_[k];
    assert(r < m);
    const i_t col_start = S_.col_start[k];
    const i_t col_end   = S_.col_start[k + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      rhs[S_.i[p]] += rhs[r] * S_.x[p];
    }
  }
  // L0'*y = c
  // TODO: handle a sparse rhs
  dual_simplex::lower_triangular_transpose_solve(L0_, rhs);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::scatter_into_workspace(const sparse_vector_t<i_t, f_t>& in) const
{
  const i_t m = L0_.m;
  // scatter pattern into xi_workspace_
  i_t nz = in.i.size();
  for (i_t k = 0; k < nz; ++k) {
    const i_t i          = in.i[k];
    xi_workspace_[i]     = 1;
    xi_workspace_[m + k] = i;
  }
  // scatter values into x_workspace_
  for (i_t k = 0; k < nz; ++k) {
    x_workspace_[in.i[k]] = in.x[k];
  }
  return nz;
}

template <typename i_t, typename f_t>
void basis_update_t<i_t, f_t>::gather_into_sparse_vector(i_t nz,
                                                         sparse_vector_t<i_t, f_t>& out) const
{
  const i_t m = L0_.m;
  out.i.clear();
  out.x.clear();
  out.i.resize(nz);
  out.x.resize(nz);
  for (i_t k = 0; k < nz; ++k) {
    const i_t i          = xi_workspace_[m + k];
    out.i[k]             = i;
    out.x[k]             = x_workspace_[i];
    xi_workspace_[m + k] = 0;
    xi_workspace_[i]     = 0;
    x_workspace_[i]      = 0.0;
  }
}

template <typename i_t, typename f_t>
void basis_update_t<i_t, f_t>::solve_to_sparse_vector(i_t top, sparse_vector_t<i_t, f_t>& out) const
{
  const i_t m = L0_.m;
  out.n       = m;
  out.i.clear();
  out.x.clear();
  const i_t nz = m - top;
  out.x.resize(nz);
  out.i.resize(nz);
  i_t k = 0;
  for (i_t p = top; p < m; ++p) {
    const i_t i      = xi_workspace_[p];
    out.i[k]         = i;
    out.x[k]         = x_workspace_[i];
    x_workspace_[i]  = 0.0;
    xi_workspace_[p] = 0;
    k++;
  }
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::l_transpose_solve(sparse_vector_t<i_t, f_t>& rhs) const
{
  // L = L0*R1^{-1}* ... * Rk^{-1}
  // L' = Rk^{-T} * Rk-1^{-T} * ... * R2^{-T} * R1^{-T} * L0^T
  // L'*y = c
  // Rk^{-T} * Rk-1^{-T} * ... * R2^{-T} * R1^{-T} * L0^T * y = c
  // L0^T * y = cprime = R1^1 * ... * Rk^T * c
  const i_t m = L0_.m;

  i_t nz = 0;

#ifdef CHECK_UPDATES
  std::vector<f_t> multiply;
  rhs.to_dense(multiply);
  for (i_t k = 0; k < 2 * m; ++k) {
    if (xi_workspace_[k]) { printf("xi workspace %d %d\n", k, xi_workspace_[k]); }
  }
#endif

  if (num_updates_ > 0) { nz = scatter_into_workspace(rhs); }

  for (i_t k = num_updates_ - 1; k >= 0; --k) {
    const i_t r = pivot_indices_[k];
    assert(r < m);
    const i_t col_start = S_.col_start[k];
    const i_t col_end   = S_.col_start[k + 1];
    if (xi_workspace_[r]) {
      for (i_t p = col_start; p < col_end; ++p) {
        // rhs.x[S_.i[p]] += rhs.x[r] * S_.x[p];
        if (!xi_workspace_[S_.i[p]]) {
          xi_workspace_[S_.i[p]] = 1;
          xi_workspace_[m + nz]  = S_.i[p];
          nz++;
        }
        x_workspace_[S_.i[p]] += x_workspace_[r] * S_.x[p];
      }
    }
#ifdef CHECK_UPDATES
    for (i_t p = col_start; p < col_end; ++p) {
      multiply[S_.i[p]] += multiply[r] * S_.x[p];
    }
#endif
  }

  // Gather into rhs
  if (num_updates_ > 0) {
    gather_into_sparse_vector(nz, rhs);

    rhs.sort();

#ifdef CHECK_UPDATES
    std::vector<f_t> rhs_dense;
    rhs.to_dense(rhs_dense);
    for (i_t k = 0; k < m; ++k) {
      if (std::abs(rhs_dense[k] - multiply[k]) > 1e-6) {
        printf("rhs dense/multiply error %d %e %e\n", k, rhs_dense[k], multiply[k]);
      }
    }
#endif
  }

  // L0^T * y = cprime
#ifdef CHECK_LOWER_TRANSPOSE_SOLVE
  std::vector<f_t> cprime_dense;
  rhs.to_dense(cprime_dense);
#endif

  i_t top = sparse_triangle_solve<i_t, f_t, false>(
    rhs, std::nullopt, xi_workspace_, L0_transpose_, x_workspace_.data());
  solve_to_sparse_vector(top, rhs);  // Uses xi_workspace_ and x_workspace_ to fill rhs

#ifdef CHECK_LOWER_TRANSPOSE_SOLVE
  std::vector<f_t> y_dense;
  rhs.to_dense(y_dense);

  std::vector<f_t> residual = cprime_dense;
  matrix_transpose_vector_multiply(L0_, 1.0, y_dense, -1.0, residual);
  const f_t L0_solve_error = vector_norm_inf<i_t, f_t>(residual);
  if (L0_solve_error > 1e-6) { printf("L0 solve error %e\n", L0_solve_error); }

#endif
  return 0;
}

template <typename i_t, typename f_t>
f_t basis_update_t<i_t, f_t>::update_lower(const std::vector<i_t>& sind,
                                           const std::vector<f_t>& sval,
                                           i_t leaving)
{
  f_t norm_s = vector_norm_inf<i_t, f_t>(sval);
  if (norm_s > 0) {
    // Currently we have S_.col_start[0..num_updates]
    const i_t current_nz = S_.col_start[num_updates_];
    const i_t s_nz       = sind.size();
    const i_t new_nz     = current_nz + s_nz;
    S_.col_start.push_back(new_nz);
    for (i_t k = 0; k < s_nz; ++k) {
      S_.i.push_back(sind[k]);
      S_.x.push_back(sval[k]);
    }
    pivot_indices_.push_back(leaving);
    S_.col_start[num_updates_ + 1] = new_nz;
    num_updates_++;
  }
  return norm_s;
}

// x = U(q, q)\b
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::u_solve(std::vector<f_t>& x) const
{
  // Solve Q*U*Q'*x = b
  // Multiplying by Q' we have U*Q'*x = Q'*b = bprime
  // Let y = Q'*x so U*y = bprime
  // 1. Compute bprime = Q'*b
  // 2. Solve for y such that U*y = bprime
  // 3. Compute Q*y = x
  const i_t m = U_.m;
  std::vector<f_t> bprime(m);
  inverse_permute_vector(col_permutation_, x, bprime);

#ifdef CHECK_UPPER_SOLVE
  std::vector<f_t> residual = bprime;
#endif

  dual_simplex::upper_triangular_solve(U_, bprime);

#ifdef CHECK_UPPER_SOLVE
  matrix_vector_multiply(U_, 1.0, bprime, -1.0, residual);
  printf("|| U0 * y - bprime || %e\n", vector_norm_inf(residual));
#endif

  permute_vector(col_permutation_, bprime, x);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::u_solve(sparse_vector_t<i_t, f_t>& rhs) const
{
  // Solve Q*U*Q'*x = b
  // Multiplying by Q' we have U*Q'*x = Q'*b = bprime
  // Let y = Q'*x so U*y = bprime
  // 1. Compute bprime = Q'*b
  // 2. Solve for y such that U*y = bprime
  // 3. Compute Q*y = x
  const i_t m = U_.m;
  sparse_vector_t<i_t, f_t> bprime(m, 0);
  rhs.inverse_permute_vector(col_permutation_, bprime);

  i_t top = sparse_triangle_solve<i_t, f_t, false>(
    bprime, std::nullopt, xi_workspace_, U_, x_workspace_.data());
  solve_to_sparse_vector(top, rhs);  // Uses xi_workspace_ and x_workspace_ to fill rhs

  rhs.inverse_permute_vector(inverse_col_permutation_);

  return 0;
}

// x = U'(q,q)\b
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::u_transpose_solve(std::vector<f_t>& x) const
{
  // Solve Q*U'*Q'*x = b
  // Multiplying by Q' we have U'*Q'*x = Q'*b = bprime
  // Let y = Q'*x so U'*y = bprime
  // 1. Compute bprime = Q'*b
  // 2. Solve for y such that U'*y = bprime
  // 3. Compute Q*y = x
  const i_t m = U_.m;
  std::vector<f_t> bprime(m);
  inverse_permute_vector(col_permutation_, x, bprime);
  dual_simplex::upper_triangular_transpose_solve(U_, bprime);
  permute_vector(col_permutation_, bprime, x);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::u_transpose_solve(sparse_vector_t<i_t, f_t>& rhs) const
{
  // Solve Q*U'*Q'*x = b
  // Multiplying by Q' we have U'*Q'*x = Q'*b = bprime
  // Let y = Q'*x so U'*y = bprime
  // 1. Compute bprime = Q'*b
  // 2. Solve for y such that U'*y = bprime
  // 3. Compute Q*y = x
  const i_t m = U_.m;
  sparse_vector_t<i_t, f_t> bprime(1, 0);
#ifdef CHECK_PERMUTATION
  std::vector<f_t> rhs_dense(m);
  rhs.to_dense(rhs_dense);
#endif
  rhs.inverse_permute_vector(col_permutation_, bprime);
#ifdef CHECK_PERMUTATION
  std::vector<f_t> bprime_dense;
  bprime.to_dense(bprime_dense);
  std::vector<f_t> rhs_dense_permuted(m);
  inverse_permute_vector(col_permutation_, rhs_dense, rhs_dense_permuted);
  for (i_t k = 0; k < m; ++k) {
    if (std::abs(bprime_dense[k] - rhs_dense_permuted[k]) > 1e-6) {
      printf("u_transpose inverse permutation error %d %e %e\n",
             k,
             bprime_dense[k],
             rhs_dense_permuted[k]);
    }
  }
#endif

#ifdef CHECK_WORKSPACE
  for (i_t k = 0; k < 2 * m; ++k) {
    if (xi_workspace_[k]) {
      printf("before Utranspose m %d solve xi workspace %d %d\n", m, k, xi_workspace_[k]);
    }
  }
#endif

  // U'*y = bprime
  i_t top = sparse_triangle_solve<i_t, f_t, true>(
    bprime, std::nullopt, xi_workspace_, U_transpose_, x_workspace_.data());
  solve_to_sparse_vector(top, rhs);  // Uses xi_workspace_ and x_workspace_ to fill rhs

#ifdef CHECK_WORKSPACE
  for (i_t k = 0; k < 2 * m; ++k) {
    if (xi_workspace_[k]) {
      printf(
        "after Utranspose m %d top %d solve xi workspace %d %d\n", m, top, k, xi_workspace_[k]);
    }
  }
#endif

#ifdef CHECK_PERMUTATION
  std::vector<f_t> rhs_dense2;
  rhs.to_dense(rhs_dense2);
#endif

  // Q*y = x
  rhs.inverse_permute_vector(inverse_col_permutation_);
#ifdef CHECK_PERMUTATION
  rhs.to_dense(rhs_dense_permuted);
  std::vector<f_t> rhs_dense_permuted2(m);
  permute_vector(col_permutation_, rhs_dense2, rhs_dense_permuted2);
  bool found_error = false;
  for (i_t k = 0; k < m; ++k) {
    if (std::abs(rhs_dense_permuted[k] - rhs_dense_permuted2[k]) > 1e-6) {
      printf("u_transpose2 permutation error %d %e %e\n",
             k,
             rhs_dense_permuted[k],
             rhs_dense_permuted2[k]);
      found_error = true;
    }
  }
  if (found_error) {
    for (i_t k = 0; k < m; ++k) {
      printf("%d (sparse -> permuted -> dense) %e (sparse -> dense -> permuted)%e\n",
             k,
             rhs_dense_permuted[k],
             rhs_dense_permuted2[k]);
    }
    for (i_t k = 0; k < rhs.i.size(); ++k) {
      printf("%d rhs sparse %d %e\n", k, rhs.i[k], rhs.x[k]);
    }
    for (i_t k = 0; k < m; ++k) {
      if (rhs_dense_permuted[k] != 0.0) {
        printf("%d rhs dense permuted %e\n", k, rhs_dense_permuted[k]);
      }
    }
    for (i_t k = 0; k < m; ++k) {
      if (rhs_dense2[k] != 0.0) { printf("%d rhs dense2 %e\n", k, rhs_dense2[k]); }
    }
    printf("col permutation %d rhs dense 2 %d rhs dense permuted %d\n",
           col_permutation_.size(),
           rhs_dense2.size(),
           rhs_dense_permuted.size());
    for (i_t k = 0; k < col_permutation_.size(); ++k) {
      printf("%d col permutation %d\n", k, col_permutation_[k]);
    }
    for (i_t k = 0; k < m; ++k) {
      printf("%d col permutation inverse %d\n", k, inverse_col_permutation_[k]);
    }
  }
#endif
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::index_map(i_t r) const
{
  // Q' * e_r = e_t
  // w = Q' * e_r = e_r(qinv)
  // w(t) = 1
  // If qinv(t) == r -> w(t) = 1
  const i_t n = inverse_col_permutation_.size();
  for (i_t k = 0; k < n; ++k) {
    if (inverse_col_permutation_[k] == r) { return k; }
  }
  return -1;
}

template <typename i_t, typename f_t>
f_t basis_update_t<i_t, f_t>::u_diagonal(i_t j) const
{
  const i_t col_end = U_.col_start[j + 1] - 1;
  assert(U_.i[col_end] == j);
  return U_.x[col_end];
}

// Ensures that the diagonal element U(j, j) is the last element in column j
// This is necessary for solves with U and U^T
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::place_diagonals()
{
  const i_t n = U_.n;
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = U_.col_start[j];
    const i_t col_end   = U_.col_start[j + 1] - 1;
    if (U_.i[col_end] == j) { continue; }
    bool found_diag = false;
    for (i_t p = col_start; p < col_end; ++p) {
      if (U_.i[p] == j) {
        // Swap with the last element in the column
        const i_t tmp_i = U_.i[col_end];
        const f_t tmp_x = U_.x[col_end];
        U_.i[col_end]   = U_.i[p];
        U_.x[col_end]   = U_.x[p];
        U_.i[p]         = tmp_i;
        U_.x[p]         = tmp_x;
        found_diag      = true;
        break;
      }
    }
    assert(found_diag);
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::update_upper(const std::vector<i_t>& ind,
                                           const std::vector<f_t>& baru,
                                           i_t t)
{
  const i_t n = U_.n;
  if (t != (n - 1)) {
    // remove U(t, :)
    U_.remove_row(t);

    // remove U(:, t)
    U_.remove_column(t);
    U_.n = n - 1;

    // place diagonals
    place_diagonals();
    U_.n = n;

    // Update q
    // Qbar = Q * Pt
    std::vector<i_t> pt(n);
    for (i_t k = 0; k < t; ++k) {
      pt[k] = k;
    }
    for (i_t k = t; k < n - 1; ++k) {
      pt[k] = k + 1;
    }
    pt[n - 1] = t;
    std::vector<i_t> ptinv(n);
    inverse_permutation(pt, ptinv);
    for (i_t k = 0; k < n; ++k) {
      col_permutation_[k] = ptinv[col_permutation_[k]];
    }
    inverse_permutation(col_permutation_, inverse_col_permutation_);
  }

  // Insert at U(:, n)
  const i_t current_col_start = U_.col_start[n - 1];
  const i_t new_col_nz        = ind.size();
  const i_t new_nz            = current_col_start + new_col_nz;
  if (new_nz > U_.i.size()) { U_.reallocate(2 * new_nz); }
  i_t p             = current_col_start;
  bool has_diagonal = false;
  for (i_t k = 0; k < new_col_nz; ++k) {
    const i_t i = ind[k];
    const f_t x = baru[k];
    if (i != n - 1) {
      U_.i[p] = i;
      U_.x[p] = x;
      p++;
    } else {
      U_.i[new_nz - 1] = i;
      U_.x[new_nz - 1] = x;
      has_diagonal     = true;
    }
  }
  assert(has_diagonal);
  U_.col_start[n] = new_nz;

  // Check to ensure that U remains upper triangular
#ifdef CHECK_UPPER_TRIANGULAR
  for (i_t k = 0; k < n; ++k) {
    const i_t col_start = U_.col_start[k];
    const i_t col_end   = U_.col_start[k + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      assert(U_.i[p] <= k);
    }
  }
#endif

  // Update U transpose
  U_.transpose(U_transpose_);

  return 0;
}

// Takes in utilde such that L*utilde = abar, where abar is the column to add to the basis
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::update(std::vector<f_t>& utilde, i_t leaving_index)
{
  // Solve L*utilde = abar
  // TODO: We should already have utilde from computing delta_x update
  // TODO: Take into account sparsity of abar
  const i_t m = L0_.m;
#ifdef RECONSTRUCT_UTILDE
  std::vector<f_t> utilde(m);
  permute_vector(row_permutation_, abar, utilde);

  l_solve(utilde);
#endif

  // ubar = Q'*utilde
  std::vector<f_t> ubar(m);
  inverse_permute_vector(col_permutation_, utilde, ubar);

  // Find t
  const i_t t = index_map(leaving_index);
  assert(t >= 0);

  // Find delta = U(t, t)
  const f_t delta = u_diagonal(t);

  // Solve U'*w = delta*et
  std::vector<f_t> w(m);
  w[t] = delta;
  dual_simplex::upper_triangular_transpose_solve(U_, w);
#ifdef PARANOID
  {
    // Compute the residual of the solve
    std::vector<f_t> residual(m);
    residual[t] = delta;
    matrix_transpose_vector_multiply(U_, 1.0, w, -1.0, residual);
    printf("|| U'*w - delta*et|| %e\n", vector_norm_inf(residual));
  }
#endif

  assert(w[t] == 1.0);

  bool update_L = false;
  for (i_t k = t + 1; k < m; ++k) {
    if (w[k] != 0.0) {
      update_L = true;
      break;
    }
  }

  // Set deltabar = w'*ubar
  const f_t deltabar = update_L ? dot<i_t, f_t>(w, ubar) : ubar[t];
  assert(std::abs(deltabar) > 0);
  std::vector<f_t> baru(m);
  for (i_t k = 0; k < t; ++k) {
    baru[k] = ubar[k];
  }
  for (i_t k = t; k < m - 1; ++k) {
    baru[k] = ubar[k + 1];
  }
  baru[m - 1] = deltabar;
  std::vector<i_t> baru_ind;
  std::vector<f_t> baru_val;
  for (i_t k = 0; k < m; ++k) {
    if (baru[k] != 0.0) {
      baru_ind.push_back(k);
      baru_val.push_back(baru[k]);
    }
  }

  std::vector<f_t> d(m);
  d    = w;
  d[t] = 0.0;
  // dtilde^T = d^T Q^T -> dtilde = Q*d
  std::vector<f_t> dtilde(m);
  permute_vector(col_permutation_, d, dtilde);

  update_upper(baru_ind, baru_val, t);
  f_t norm_s = 0.0;
  if (t != (m - 1)) {
    std::vector<i_t> sind;
    std::vector<f_t> sval;
    for (i_t i = 0; i < m; ++i) {
      if (dtilde[i] != 0.0) {
        sind.push_back(i);
        sval.push_back(dtilde[i]);
      }
    }
    norm_s = update_lower(sind, sval, leaving_index);
  }

#ifdef CHECK_ABAR
  {
    sparse_matrix_t abar_test(m, 1, 1);
    const Int nz           = lower_triangular_multiply(U_, m - 1, abar_test, 1);
    abar_test.col_start[1] = nz;
    std::vector<f_t> abar_test_dense_unperm(m);
    const Int col_nz = abar_test.col_start[1];
    for (Int p = 0; p < col_nz; ++p) {
      assert(abar_test.i[p] < m);
      abar_test_dense_unperm[abar_test.i[p]] = abar_test.x[p];
    }
    std::vector<f_t> abar_test_dense = abar_test_dense_unperm;
    inverse_permute_vector(row_permutation_, abar_test_dense_unperm, abar_test_dense);
    f_t max_err = 0;
    for (Int k = 0; k < m; ++k) {
      const f_t err = std::abs(abar_test_dense[k] - abar[k]);
      if (err > 1e-4) {
        printf("error abar %d %e recover %e orig %e\n", k, err, abar_test_dense[k], abar[k]);
      }
      max_err = std::max(max_err, err);
    }
    assert(max_err < 1e-3);
  }
#endif

  i_t should_refactor = 0;
  if (norm_s > 1e5) { should_refactor = 1; }
  return should_refactor;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::multiply_lu(csc_matrix_t<i_t, f_t>& out)
{
  const i_t m = L0_.m;
  out.col_start.resize(m + 1);
  assert(out.m == m);
  const i_t nz_estimate = L0_.col_start[m] + U_.col_start[m];
#ifdef PRINT_PERMUTATIONS
  printf("Nz estimate %d m %d num updates %d\n", nz_estimate, m, num_updates_);
  printf("q = [");
  for (i_t k = 0; k < m; ++k) {
    printf("%d ", col_permutation_[k]);
  }
  printf("];\n");
  // PrintMatrix(L0_);
  printf("p = [");
  for (i_t k = 0; k < m; ++k) {
    printf("%d ", row_permutation_[k]);
  }
  printf("];\n");
#endif
  out.reallocate(nz_estimate);

  // out(:, j) = L * Q * U * Q' e_j
  // out(:, j) = L*U(q, q(j))
  i_t nz = 0;
  for (i_t j = 0; j < m; ++j) {
    out.col_start[j] = nz;
    const i_t k      = col_permutation_[j];
    nz               = lower_triangular_multiply(U_, k, out, j);
  }
  out.col_start[m] = nz;

  // L*U = P*B
  // P'*L*U = B
  for (i_t k = 0; k < m; ++k) {
    const i_t col_start = out.col_start[k];
    const i_t col_end   = out.col_start[k + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      out.i[p] = row_permutation_[out.i[p]];
    }
  }
  return 0;
}

// out(:, out_col) = L*in(:, in_col)
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::lower_triangular_multiply(const csc_matrix_t<i_t, f_t>& in,
                                                        i_t in_col,
                                                        csc_matrix_t<i_t, f_t>& out,
                                                        i_t out_col) const
{
  const i_t m = in.m;
  // L = L0 * R1^{-1} * R2^{-1} * ... * Rk^{-1}
  //
  // where Ri = I + e_r d^T
  //       Ri^{-1} = I - e_r d^T

  // x = U(q, q(j))
  std::vector<i_t> sind;
  std::vector<f_t> sval;
  const i_t in_col_start = in.col_start[in_col];
  const i_t in_col_end   = in.col_start[in_col + 1];
  std::vector<f_t> sbuffer(m);
  for (i_t p = in_col_start; p < in_col_end; ++p) {
    sbuffer[inverse_col_permutation_[in.i[p]]] = in.x[p];
  }
  for (i_t k = 0; k < m; ++k) {
    if (sbuffer[k] != 0) {
      sind.push_back(k);
      sval.push_back(sbuffer[k]);
    }
  }

  for (i_t k = num_updates_ - 1; k >= 0; --k) {
    // Ri^{-1} * x = (I - e_r d^T) * x = x - e_r d^T x
    const i_t r = pivot_indices_[k];
    f_t dot     = sparse_dot(sind, sval, S_, k);
    if (dot == 0.0) { continue; }
#ifdef DEBUG_MULTIPLY
    for (i_t p = 0; p < sind.size(); ++p) {
      printf("s %d %e\n", sind[p], sval[p]);
    }
    printf("S col start %d %d\n", S_.col_start[k], S_.col_start[k + 1]);
    for (i_t p = S_.col_start[k]; p < S_.col_start[k + 1]; ++p) {
      printf("S %d %d %e\n", k, S_.i[p], S_.x[p]);
    }
#endif
    bool fill = true;
    for (i_t p = 0; p < sind.size(); ++p) {
      if (sind[p] == r) {
        sval[p] -= dot;
        fill = false;
        break;
      }
    }
    if (fill) {
      std::vector<f_t> work2(m);
      sind.push_back(r);
      sval.push_back(-dot);

      for (i_t p = 0; p < sind.size(); ++p) {
        work2[sind[p]] = sval[p];
      }
      sind.clear();
      sval.clear();
      for (i_t i = 0; i < m; ++i) {
        if (work2[i] != 0.0) {
          sind.push_back(i);
          sval.push_back(work2[i]);
        }
      }
    }
    // assert(fill == false);
  }

  std::vector<f_t> workspace(m);
  const i_t nx = sind.size();
  for (i_t k = 0; k < nx; ++k) {
    const i_t j  = sind[k];
    const f_t x  = sval[k];
    workspace[j] = x;
  }
  std::vector<f_t> workspace2(m);
  matrix_vector_multiply(L0_, 1.0, workspace, 0.0, workspace2);
  workspace = workspace2;

  i_t col_nz = 0;
  for (i_t i = 0; i < m; ++i) {
    if (workspace[i] != 0.0) { col_nz++; }
  }
  const i_t nz     = out.col_start[out_col];
  const i_t new_nz = nz + col_nz;
  if (out.i.size() < new_nz) { out.reallocate(new_nz); }

  i_t p = nz;
  for (i_t i = 0; i < m; ++i) {
    if (workspace[i] != 0.0) {
      out.i[p] = i;
      out.x[p] = workspace[i];
      p++;
    }
  }
  assert(p == new_nz);
  return new_nz;
}

template <typename i_t, typename f_t>
void basis_update_mpf_t<i_t, f_t>::gather_into_sparse_vector(i_t nz,
                                                             sparse_vector_t<i_t, f_t>& out) const
{
  const i_t m = L0_.m;
  out.i.clear();
  out.x.clear();
  out.i.reserve(nz);
  out.x.reserve(nz);
  const f_t zero_tol = 1e-13;
  for (i_t k = 0; k < nz; ++k) {
    const i_t i = xi_workspace_[m + k];
    if (std::abs(x_workspace_[i]) > zero_tol) {
      out.i.push_back(i);
      out.x.push_back(x_workspace_[i]);
    }
    xi_workspace_[m + k] = 0;
    xi_workspace_[i]     = 0;
    x_workspace_[i]      = 0.0;
  }
}

template <typename i_t, typename f_t>
void basis_update_mpf_t<i_t, f_t>::solve_to_workspace(i_t top) const
{
  const i_t m = L0_.m;
  i_t nz      = 0;
  for (i_t p = top; p < m; ++p) {
    const i_t i           = xi_workspace_[p];
    xi_workspace_[m + nz] = i;
    xi_workspace_[p]      = 0;
    nz++;
  }
  for (i_t k = 0; k < nz; ++k) {
    const i_t i      = xi_workspace_[m + k];
    xi_workspace_[i] = 1;
  }
}

template <typename i_t, typename f_t>
void basis_update_mpf_t<i_t, f_t>::solve_to_sparse_vector(i_t top,
                                                          sparse_vector_t<i_t, f_t>& out) const
{
  const i_t m  = L0_.m;
  out.n        = m;
  const i_t nz = m - top;
  out.x.clear();
  out.i.clear();
  out.x.reserve(nz);
  out.i.reserve(nz);
  i_t k              = 0;
  const f_t zero_tol = 1e-13;
  for (i_t p = top; p < m; ++p) {
    const i_t i = xi_workspace_[p];
    if (std::abs(x_workspace_[i]) > zero_tol) {
      out.i.push_back(i);
      out.x.push_back(x_workspace_[i]);
    }
    x_workspace_[i]  = 0.0;
    xi_workspace_[p] = 0;
    k++;
  }
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::scatter_into_workspace(const sparse_vector_t<i_t, f_t>& in) const
{
  const i_t m = L0_.m;
  // scatter pattern into xi_workspace_
  i_t nz = in.i.size();
  for (i_t k = 0; k < nz; ++k) {
    const i_t i          = in.i[k];
    xi_workspace_[i]     = 1;
    xi_workspace_[m + k] = i;
  }
  // scatter values into x_workspace_
  for (i_t k = 0; k < nz; ++k) {
    x_workspace_[in.i[k]] = in.x[k];
  }
  return nz;
}

template <typename i_t, typename f_t>
void basis_update_mpf_t<i_t, f_t>::grow_storage(i_t nz, i_t& S_start, i_t& S_nz)
{
  const i_t last_S_col     = num_updates_ * 2;
  const i_t new_last_S_col = last_S_col + 2;
  if (new_last_S_col >= S_.col_start.size()) {
    S_.col_start.resize(new_last_S_col + refactor_frequency_);
  }
  S_nz = S_.col_start[last_S_col];
  if (S_nz + nz > S_.i.size()) {
    S_.i.resize(std::max(2 * S_nz, S_nz + nz));
    S_.x.resize(std::max(2 * S_nz, S_nz + nz));
  }
  S_start = last_S_col;
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::nonzeros(const std::vector<f_t>& x) const
{
  i_t nz        = 0;
  const i_t xsz = x.size();
  for (i_t i = 0; i < xsz; ++i) {
    if (x[i] != 0.0) { nz++; }
  }
  return nz;
}

// dot = S(:, col)' * x
template <typename i_t, typename f_t>
f_t basis_update_mpf_t<i_t, f_t>::dot_product(i_t col, const std::vector<f_t>& x) const
{
  f_t dot             = 0.0;
  const i_t col_start = S_.col_start[col];
  const i_t col_end   = S_.col_start[col + 1];
  for (i_t p = col_start; p < col_end; ++p) {
    const i_t i = S_.i[p];
    dot += S_.x[p] * x[i];
  }
  return dot;
}

// dot = S(:, col)' * x
template <typename i_t, typename f_t>
f_t basis_update_mpf_t<i_t, f_t>::dot_product(i_t col,
                                              const std::vector<i_t>& mark,
                                              const std::vector<f_t>& x) const
{
  f_t dot             = 0.0;
  const i_t col_start = S_.col_start[col];
  const i_t col_end   = S_.col_start[col + 1];
  for (i_t p = col_start; p < col_end; ++p) {
    const i_t i = S_.i[p];
    if (mark[i]) { dot += S_.x[p] * x[i]; }
  }
  return dot;
}

// x <- x + theta * S(:, col)
template <typename i_t, typename f_t>
void basis_update_mpf_t<i_t, f_t>::add_sparse_column(const csc_matrix_t<i_t, f_t>& S,
                                                     i_t col,
                                                     f_t theta,
                                                     std::vector<f_t>& x) const
{
  const i_t col_start = S.col_start[col];
  const i_t col_end   = S.col_start[col + 1];
  for (i_t p = col_start; p < col_end; ++p) {
    const i_t i = S.i[p];
    x[i] += theta * S.x[p];
  }
}

template <typename i_t, typename f_t>
void basis_update_mpf_t<i_t, f_t>::add_sparse_column(const csc_matrix_t<i_t, f_t>& S,
                                                     i_t col,
                                                     f_t theta,
                                                     std::vector<i_t>& mark,
                                                     i_t& nz,
                                                     std::vector<f_t>& x) const
{
  const i_t m         = L0_.m;
  const i_t col_start = S.col_start[col];
  const i_t col_end   = S.col_start[col + 1];
  for (i_t p = col_start; p < col_end; ++p) {
    const i_t i = S.i[p];
    if (!mark[i]) {
      // Fill occured
      mark[i]      = 1;
      mark[m + nz] = i;
      nz++;
    }
    x[i] += theta * S.x[p];
  }
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::b_transpose_solve(const std::vector<f_t>& rhs,
                                                    std::vector<f_t>& solution) const
{
  std::vector<f_t> UTsol;
  return b_transpose_solve(rhs, solution, UTsol);
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::b_transpose_solve(const std::vector<f_t>& rhs,
                                                    std::vector<f_t>& solution,
                                                    std::vector<f_t>& UTsol) const
{
  const i_t m = L0_.m;
  // P*B = L*U
  // B'*P' = U'*L'
  // We want to solve
  // B'*y = c
  // Let y = P'*w
  // B'*y = B'*P'*w = U'*L'*w = c
  // 1. Solve U'*r = c for r
  // 2. Solve L'*w = r for w
  // 3. Compute y = P'*w

  // Solve for r such that U'*r = c
  std::vector<f_t> r = rhs;
  u_transpose_solve(r);
  UTsol = r;

  // Solve for w such that L'*w = r
  l_transpose_solve(r);

  // Compute y = P'*w
  inverse_permute_vector(row_permutation_, r, solution);

  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::b_transpose_solve(const sparse_vector_t<i_t, f_t>& rhs,
                                                    sparse_vector_t<i_t, f_t>& solution) const
{
  sparse_vector_t<i_t, f_t> UTsol(1, 0);
  return b_transpose_solve(rhs, solution, UTsol);
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::b_transpose_solve(const sparse_vector_t<i_t, f_t>& rhs,
                                                    sparse_vector_t<i_t, f_t>& solution,
                                                    sparse_vector_t<i_t, f_t>& UTsol) const
{
  // Solve for r such that U'*r = c

  bool use_hypersparse = false;
  const f_t input_size = static_cast<f_t>(rhs.i.size());
  estimate_solution_density(input_size, sum_U_transpose_, num_calls_U_transpose_, use_hypersparse);
  if (use_hypersparse) {
    solution = rhs;
    u_transpose_solve(solution);
  } else {
    std::vector<f_t> solution_dense;
    rhs.to_dense(solution_dense);
    u_transpose_solve(solution_dense);
    solution.from_dense(solution_dense);
  }
  UTsol = solution;
  sum_U_transpose_ += static_cast<f_t>(solution.i.size()) / input_size;

#ifdef CHECK_U_TRANSPOSE_SOLVE
  std::vector<f_t> UTsol_dense;
  UTsol.to_dense(UTsol_dense);
  std::vector<f_t> rhs_dense;
  rhs.to_dense(rhs_dense);

  matrix_transpose_vector_multiply(U0_, 1.0, UTsol_dense, -1.0, rhs_dense);
  if (vector_norm_inf<i_t, f_t>(rhs_dense) > 1e-10) {
    printf("B transpose solve U transpose residual %e\n", vector_norm_inf<i_t, f_t>(rhs_dense));
  }
#endif

  // Solve for w such that L'*w = r
#ifdef CHECK_L_TRANSPOSE_SOLVE
  std::vector<f_t> r_dense;
  solution.to_dense(r_dense);
#endif
  const f_t rhs_size = static_cast<f_t>(solution.i.size());
  estimate_solution_density(rhs_size, sum_L_transpose_, num_calls_L_transpose_, use_hypersparse);
  if (use_hypersparse) {
    l_transpose_solve(solution);
  } else {
    std::vector<f_t> solution_dense;
    solution.to_dense(solution_dense);
    l_transpose_solve(solution_dense);
    solution.from_dense(solution_dense);
  }
  sum_L_transpose_ += static_cast<f_t>(solution.i.size()) / rhs_size;

#ifdef CHECK_L_TRANSPOSE_SOLVE
  std::vector<f_t> solution_dense;
  solution.to_dense(solution_dense);
  l_transpose_multiply(solution_dense);
  f_t max_error = 0.0;
  for (i_t k = 0; k < L0_.m; ++k) {
    if (std::abs(solution_dense[k] - r_dense[k]) > 1e-4) {
      printf(
        "B transpose solve L transpose solve error %e: index %d multiply %e rhs %e. update %d. use "
        "hypersparse %d\n",
        std::abs(solution_dense[k] - r_dense[k]),
        k,
        solution_dense[k],
        r_dense[k],
        num_updates_,
        use_hypersparse);
    }

    max_error = std::max(max_error, std::abs(solution_dense[k] - r_dense[k]));
  }
  if (max_error > 1e-4) { printf("B transpose solve L transpose solve residual %e\n", max_error); }
#endif
  // Compute y = P'*w
  solution.inverse_permute_vector(row_permutation_);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::u_transpose_solve(std::vector<f_t>& rhs) const
{
  total_dense_U_transpose_++;
  dual_simplex::upper_triangular_transpose_solve(U0_, rhs);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::u_transpose_solve(sparse_vector_t<i_t, f_t>& rhs) const
{
  total_sparse_U_transpose_++;
  // U0'*x = y
  // Solve U0'*x0 = y
  i_t top = dual_simplex::sparse_triangle_solve<i_t, f_t, true>(
    rhs, std::nullopt, xi_workspace_, U0_transpose_, x_workspace_.data());
  solve_to_sparse_vector(top, rhs);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::l_transpose_solve(std::vector<f_t>& rhs) const
{
  total_dense_L_transpose_++;
  // L = L0 * T0 * T1 * ... * T_{num_updates_ - 1}
  // L' = T_{num_updates_ - 1}^T * T_{num_updates_ - 2}^T * ... * T0^T * L0^T
  // L'*x = b
  // L0^T *x = T_0^-T * T_1^-T * ... * T_{num_updates_ - 1}^-T * b = b'

  const f_t zero_tol = 1e-13;
  // Compute b'
  for (i_t k = num_updates_ - 1; k >= 0; --k) {
    // T_k^{-T} = ( I - v u^T/(1 + u^T v))
    // T_k^{-T} * b = b - v * (u^T * b) / (1 + u^T * v) = b - theta * v, theta = u^T b / mu

    const i_t u_col = 2 * k;
    const i_t v_col = 2 * k + 1;
    const f_t mu    = mu_values_[k];

    // dot = u^T * b
    f_t dot         = dot_product(u_col, rhs);
    const f_t theta = dot / mu;

    if (std::abs(theta) > zero_tol) { add_sparse_column(S_, v_col, -theta, rhs); }
  }

  // Solve for x such that L0^T * x = b'
  dual_simplex::lower_triangular_transpose_solve(L0_, rhs);

  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::l_transpose_solve(sparse_vector_t<i_t, f_t>& rhs) const
{
  total_sparse_L_transpose_++;
  const i_t m = L0_.m;
  // L'*x = b
  // L0^T * x = T_0^-T * T_1^-T * ... * T_{num_updates_ - 1}^-T * b = b'

  scatter_into_workspace(rhs);
  i_t nz = rhs.i.size();

#ifdef CHECK_MULTIPLY
  std::vector<f_t> rhs_dense_0;
  rhs.to_dense(rhs_dense_0);
#endif
  const f_t zero_tol = 1e-13;
  // Compute b'
  for (i_t k = num_updates_ - 1; k >= 0; --k) {
    // T_k^{-T} = ( I - v u^T/(1 + u^T v))
    // T_k^{-T} * b = b - v * (u^T * b) / (1 + u^T * v) = b - theta * v, theta = u^T b / mu

    const i_t u_col = 2 * k;
    const i_t v_col = 2 * k + 1;
    const f_t mu    = mu_values_[k];

    // dot = u^T * b
    f_t dot = dot_product(u_col, xi_workspace_, x_workspace_);

#ifdef CHECK_MULTIPLY
    f_t dot_check = 0.0;
    for (i_t p = S_.col_start[u_col]; p < S_.col_start[u_col + 1]; ++p) {
      const i_t i = S_.i[p];
      dot_check += S_.x[p] * rhs_dense_0[i];
    }
    if (std::abs(dot - dot_check) > 1e-10) {
      printf("L transpose solve dot erorr: index %d dot %e dot check %e\n", k, dot, dot_check);
    }
#endif

    const f_t theta = dot / mu;
    if (std::abs(theta) > zero_tol) {
      add_sparse_column(S_, v_col, -theta, xi_workspace_, nz, x_workspace_);
    }

#ifdef CHECK_MULTIPLY
    for (i_t p = S_.col_start[v_col]; p < S_.col_start[v_col + 1]; ++p) {
      const i_t i = S_.i[p];
      rhs_dense_0[i] -= theta * S_.x[p];
    }
#endif
  }

#ifdef CHECK_MULTIPLY
  for (i_t i = 0; i < m; ++i) {
    if (std::abs(rhs_dense_0[i] - x_workspace_[i]) > 1e-9) {
      printf("L transpose solve multiply error %e index %d sparse %e dense %e\n",
             std::abs(rhs_dense_0[i] - x_workspace_[i]),
             i,
             x_workspace_[i],
             rhs_dense_0[i]);
    }
  }
#endif

  sparse_vector_t<i_t, f_t> b(m, nz);
  gather_into_sparse_vector(nz, b);
  i_t top = dual_simplex::sparse_triangle_solve<i_t, f_t, false>(
    b, std::nullopt, xi_workspace_, L0_transpose_, x_workspace_.data());
  solve_to_sparse_vector(top, rhs);

#ifdef CHECK_SPARSE_SOLVE
  std::vector<f_t> rhs_dense;
  rhs.to_dense(rhs_dense);

  std::vector<f_t> b_dense(m, 0.0);
  for (i_t p = 0; p < nz; ++p) {
    const i_t i = b.i[p];
    b_dense[i]  = b.x[p];
  }
  matrix_vector_multiply(L0_transpose_, 1.0, rhs_dense, -1.0, b_dense);
  if (vector_norm_inf<i_t, f_t>(b_dense) > 1e-9) {
    printf("L0 transpose solve residual %e\n", vector_norm_inf<i_t, f_t>(b_dense));
  }
#endif

  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::b_solve(const std::vector<f_t>& rhs,
                                          std::vector<f_t>& solution) const
{
  const i_t m = L0_.m;
  std::vector<f_t> Lsol(m);
  return b_solve(rhs, solution, Lsol);
}

// Solve for x such that B*x = y
template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::b_solve(const std::vector<f_t>& rhs,
                                          std::vector<f_t>& solution,
                                          std::vector<f_t>& Lsol,
                                          bool need_Lsol) const
{
  const i_t m = L0_.m;
  // P*B = L*U
  // B*x = b
  // P*B*x = P*b

  permute_vector(row_permutation_, rhs, solution);

  // L*U*x = b'
  // Solve for v such that L*v = b'
#ifdef CHECK_L_SOLVE
  std::vector<f_t> rhs_permuted = solution;
#endif
  l_solve(solution);
  if (need_Lsol) { Lsol = solution; }

#ifdef CHECK_L_SOLVE
  std::vector<f_t> Lsol_check = Lsol;
  l_multiply(Lsol_check);
  f_t max_lsol_err = 0.0;
  for (i_t k = 0; k < m; ++k) {
    const f_t err = std::abs(Lsol_check[k] - rhs_permuted[k]);
    max_lsol_err  = std::max(max_lsol_err, err);
  }
  printf("B solve L multiply error %e\n", max_lsol_err);
#endif

  // Solve for x such that U*x = v
  u_solve(solution);

#ifdef CHECK_U_SOLVE
  std::vector<f_t> residual = Lsol;
  matrix_vector_multiply(U0_, 1.0, solution, -1.0, residual);
  f_t max_err = vector_norm_inf<i_t, f_t>(residual);
  printf("B solve U solve residual %e\n", max_err);
#endif
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::b_solve(const sparse_vector_t<i_t, f_t>& rhs,
                                          sparse_vector_t<i_t, f_t>& solution) const
{
  sparse_vector_t<i_t, f_t> Lsol(1, 0);
  return b_solve(rhs, solution, Lsol, false);
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::b_solve(const sparse_vector_t<i_t, f_t>& rhs,
                                          sparse_vector_t<i_t, f_t>& solution,
                                          sparse_vector_t<i_t, f_t>& Lsol,
                                          bool need_Lsol) const
{
  const i_t m = L0_.m;
  solution    = rhs;
  solution.inverse_permute_vector(inverse_row_permutation_);

#ifdef CHECK_PERMUTATION
  std::vector<f_t> permuation_rhs;
  rhs.to_dense(permuation_rhs);
  std::vector<f_t> finish_perm(m);
  permute_vector(row_permutation_, permuation_rhs, finish_perm);

  std::vector<f_t> solution_dense2;
  solution.to_dense(solution_dense2);
  for (i_t k = 0; k < m; ++k) {
    if (finish_perm[k] != solution_dense2[k]) {
      printf("B solve permutation error %e %e %d\n", finish_perm[k], solution_dense2[k], k);
    }
  }
#endif

#ifdef CHECK_L_SOLVE
  std::vector<f_t> l_solve_rhs;
  solution.to_dense(l_solve_rhs);
#endif

  bool use_hypersparse;
  const f_t input_size = static_cast<f_t>(rhs.i.size());
  estimate_solution_density(input_size, sum_L_, num_calls_L_, use_hypersparse);
  if (use_hypersparse) {
    l_solve(solution);
  } else {
    std::vector<f_t> solution_dense;
    solution.to_dense(solution_dense);
    l_solve(solution_dense);
    solution.from_dense(solution_dense);
  }
  if (need_Lsol) { Lsol = solution; }
  sum_L_ += static_cast<f_t>(solution.i.size()) / input_size;

#ifdef CHECK_L_SOLVE
  std::vector<f_t> l_solve_dense;
  Lsol.to_dense(l_solve_dense);

  l_multiply(l_solve_dense);
  f_t max_err_l_solve = 0.0;
  for (i_t k = 0; k < m; ++k) {
    const f_t err   = std::abs(l_solve_dense[k] - l_solve_rhs[k]);
    max_err_l_solve = std::max(max_err_l_solve, err);
  }
  if (max_err_l_solve > 1e-9) { printf("B solve L solve residual %e\n", max_err_l_solve); }
#endif

#ifdef CHECK_U_SOLVE
  std::vector<f_t> rhs_dense;
  solution.to_dense(rhs_dense);
#endif

  const f_t rhs_size = static_cast<f_t>(solution.i.size());
  estimate_solution_density(rhs_size, sum_U_, num_calls_U_, use_hypersparse);
  if (use_hypersparse) {
    u_solve(solution);
  } else {
    std::vector<f_t> solution_dense;
    solution.to_dense(solution_dense);
    u_solve(solution_dense);
    solution.from_dense(solution_dense);
  }
  sum_U_ += static_cast<f_t>(solution.i.size()) / rhs_size;

#ifdef CHECK_U_SOLVE
  std::vector<f_t> solution_dense;
  solution.to_dense(solution_dense);

  matrix_vector_multiply(U0_, 1.0, solution_dense, -1.0, rhs_dense);

  const f_t max_err = vector_norm_inf<i_t, f_t>(rhs_dense);
  if (max_err > 1e-9) { printf("B solve U0 solve residual %e\n", max_err); }
#endif
  return 0;
}

// Solve for x such that U*x = y
template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::u_solve(std::vector<f_t>& rhs) const
{
  total_dense_U_++;
  const i_t m = L0_.m;
  // U*x = y
  dual_simplex::upper_triangular_solve(U0_, rhs);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::u_solve(sparse_vector_t<i_t, f_t>& rhs) const
{
  total_sparse_U_++;
  const i_t m = L0_.m;
  // U*x = y

  // Solve U0*x = y
  i_t top = dual_simplex::sparse_triangle_solve<i_t, f_t, false>(
    rhs, std::nullopt, xi_workspace_, U0_, x_workspace_.data());
  solve_to_sparse_vector(top, rhs);

  return 0;
}
// Solve for x such that L*x = y
template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::l_solve(std::vector<f_t>& rhs) const
{
  total_dense_L_++;
  const i_t m = L0_.m;
  // L*x = y
  // L0 * T0 * T1 * ... * T_{num_updates_ - 1} * x = y

  // First solve L0*x0 = y
#ifdef CHECK_L0_SOLVE
  std::vector<f_t> residual = rhs;
#endif
#ifdef CHECK_L_SOLVE
  std::vector<f_t> rhs_check = rhs;
#endif
  dual_simplex::lower_triangular_solve(L0_, rhs);

#ifdef CHECK_L0_SOLVE
  matrix_vector_multiply(L0_, 1.0, rhs, -1.0, residual);
  f_t max_err = vector_norm_inf<i_t, f_t>(residual);
  printf("L solve: L0 solve residual %e\n", max_err);
#endif

  // Then T0 * T1 * ... * T_{num_updates_ - 1} * x = x0
  // Or x = T_{num_updates}^{-1} * T_1^{-1} * T_0^{-1}  x0
  const f_t zero_tol = 1e-16;  // Any higher and pilot_ja fails
  for (i_t k = 0; k < num_updates_; ++k) {
    // T = I + u*v^T
    // T^{-1} = I - u*v^T / (1 + v^T*u)
    // T^{-1} * x = x - u*v^T * x / (1 + v^T*u) = x - theta * u, theta = v^T * x / (1 + v^T*u) = v^T
    // x / mu
    const f_t mu    = mu_values_[k];
    const i_t u_col = 2 * k;
    const i_t v_col = 2 * k + 1;
    f_t dot         = dot_product(v_col, rhs);
    const f_t theta = dot / mu;

    if (std::abs(theta) > zero_tol) { add_sparse_column(S_, u_col, -theta, rhs); }
  }

#ifdef CHECK_L_SOLVE
  std::vector<f_t> inout = rhs;
  l_multiply(inout);
  f_t err_max = 0.0;
  for (i_t k = 0; k < m; ++k) {
    const f_t err = std::abs(inout[k] - rhs_check[k]);
    err_max       = std::max(err_max, err);
  }
  printf("L solve residual %e\n", err_max);
#endif

  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::l_solve(sparse_vector_t<i_t, f_t>& rhs) const
{
  total_sparse_L_++;
  const i_t m = L0_.m;
  // L*x = y
  // L0 * T0 * T1 * ... * T_{num_updates_ - 1} * x = y

  // First solve L0*x0 = y
  i_t top = dual_simplex::sparse_triangle_solve<i_t, f_t, true>(
    rhs, std::nullopt, xi_workspace_, L0_, x_workspace_.data());
  solve_to_workspace(top);  // Uses xi_workspace_ and x_workspace_ to fill rhs
  i_t nz = m - top;
  // Then T0 * T1 * ... * T_{num_updates_ - 1} * x = x0
  // Or x = T_{num_updates}^{-1} * T_1^{-1} * T_0^{-1}  x0
  const f_t zero_tol = 1e-13;
  for (i_t k = 0; k < num_updates_; ++k) {
    // T = I + u*v^T
    // T^{-1} = I - u*v^T / (1 + v^T*u)
    // T^{-1} * x = x - u*v^T * x / (1 + v^T*u) = x - theta * u, theta = v^T * x / (1 + v^T*u) = v^T
    // x / mu
    const f_t mu    = mu_values_[k];
    const i_t u_col = 2 * k;
    const i_t v_col = 2 * k + 1;

    // dot = v^T * x
    f_t dot = dot_product(v_col, xi_workspace_, x_workspace_);

    const f_t theta = dot / mu;
    if (std::abs(theta) > zero_tol) {
      add_sparse_column(S_, u_col, -theta, xi_workspace_, nz, x_workspace_);
    }
  }

  gather_into_sparse_vector(nz, rhs);

  return 0;
}

// Takes in utilde such that L*utilde = abar, where abar is the column to add to the basis
// and etilde such that U'*etilde = e_leaving
template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::update(const std::vector<f_t>& utilde,
                                         const std::vector<f_t>& etilde,
                                         i_t leaving_index)
{
  const i_t m = L0_.m;
#ifdef PRINT_NUM_UPDATES
  printf("Update: num_updates_ %d\n", num_updates_);
#endif

  // We are going to create a new matrix T = I + u*v^T
  const i_t col_start = U0_.col_start[leaving_index];
  const i_t col_end   = U0_.col_start[leaving_index + 1];
  std::vector<f_t> u  = utilde;
  // u = utilde - U0(:, leaving_index)
  add_sparse_column(U0_, leaving_index, -1.0, u);

  i_t u_nz = nonzeros(u);

  // v = etilde
  i_t v_nz = nonzeros(etilde);

  i_t nz = u_nz + v_nz;
  i_t S_start;
  i_t S_nz;
  grow_storage(nz, S_start, S_nz);
#ifdef PRINT_NZ_INFO
  printf("Update: S_start %d S_nz %d num updates %d S.n %d\n", S_start, S_nz, num_updates_, S_.n);
#endif

  i_t S_nz_start = S_nz;

  // Scatter u into S
  S_.append_column(u);

  // Scatter v into S
  S_.append_column(etilde);

  // Compute mu = 1 + v^T * u
  const f_t mu = 1.0 + sparse_dot(S_.i.data() + S_.col_start[S_start],
                                  S_.x.data() + S_.col_start[S_start],
                                  S_.col_start[S_start + 1] - S_.col_start[S_start],
                                  S_.i.data() + S_.col_start[S_start + 1],
                                  S_.x.data() + S_.col_start[S_start + 1],
                                  v_nz);

#ifdef CHECK_MU
  const f_t mu_check = 1.0 + dot<i_t, f_t>(etilde, u);
  printf("Update: mu %e mu_check %e diff %e\n", mu, mu_check, std::abs(mu - mu_check));
#endif
  mu_values_.push_back(mu);

#ifdef PRINT_MU_INFO
  printf("Update mu %e u nz %d v nz %d\n",
         mu_values_.back(),
         S_.col_start[S_start + 1] - S_.col_start[S_start],
         S_.col_start[S_start + 2] - S_.col_start[S_start + 1]);
#endif
  num_updates_++;

  return 0;
}

// Takes in utilde such that L*utilde = abar, where abar is the column to add to the basis
template <typename i_t, typename f_t>
i_t basis_update_mpf_t<i_t, f_t>::update(const sparse_vector_t<i_t, f_t>& utilde,
                                         sparse_vector_t<i_t, f_t>& etilde,
                                         i_t leaving_index)
{
  const i_t m = L0_.m;
#ifdef PRINT_NUM_UPDATES
  printf("Update: num_updates_ %d\n", num_updates_);
#endif

  // We are going to create a new matrix T = I + u*v^T
  // where u = utilde - U0(:, p) and v = etilde

  // Scatter utilde into the workspace
  i_t nz = scatter_into_workspace(utilde);

  // Subtract the column of U0 corresponding to the leaving index
  add_sparse_column(U0_, leaving_index, -1.0, xi_workspace_, nz, x_workspace_);

  // Ensure the workspace is sorted. Otherwise, the sparse dot will be incorrect.
  std::sort(xi_workspace_.begin() + m, xi_workspace_.begin() + m + nz, std::less<i_t>());

  // Gather the workspace into a column of S
  i_t S_start;
  i_t S_nz;
  grow_storage(nz + etilde.i.size(), S_start, S_nz);

  S_.append_column(nz, xi_workspace_.data() + m, x_workspace_.data());

  // Gather etilde into a column of S
  etilde.sort();  // Needs to be sorted for the sparse dot. TODO(CMM): Is etilde sorted on input?
  S_.append_column(etilde);

  // Compute mu = 1 + v^T * u
  mu_values_.push_back(1.0 + sparse_dot(S_.i.data() + S_.col_start[S_start],
                                        S_.x.data() + S_.col_start[S_start],
                                        S_.col_start[S_start + 1] - S_.col_start[S_start],
                                        S_.i.data() + S_.col_start[S_start + 1],
                                        S_.x.data() + S_.col_start[S_start + 1],
                                        S_.col_start[S_start + 2] - S_.col_start[S_start + 1]));
  // Clear the workspace
  for (i_t k = 0; k < nz; ++k) {
    const i_t i          = xi_workspace_[m + k];
    xi_workspace_[i]     = 0;
    x_workspace_[i]      = 0.0;
    xi_workspace_[m + k] = 0;
  }

#ifdef PRINT_MU_INFO
  printf("Update mu %e u nz %d v nz %d\n",
         mu_values_.back(),
         S_.col_start[S_start + 1] - S_.col_start[S_start],
         S_.col_start[S_start + 2] - S_.col_start[S_start + 1]);
#endif

  num_updates_++;

  return 0;
}

template <typename i_t, typename f_t>
void basis_update_mpf_t<i_t, f_t>::l_multiply(std::vector<f_t>& inout) const
{
  const i_t m = L0_.m;
  // L*x = y
  // L0 * T0 * T1 * ... * T_{num_updates_ - 1} * x = y

  for (i_t k = num_updates_ - 1; k >= 0; --k) {
    // T_k = ( I + u v^T)
    // T_k * b = b + u * (v^T * b) = b + theta * u, theta = v^T b
    const i_t u_col = 2 * k;
    const i_t v_col = 2 * k + 1;
    const f_t mu    = mu_values_[k];

    // dot = v^T b
    f_t dot         = dot_product(v_col, inout);
    const f_t theta = dot;
    add_sparse_column(S_, u_col, theta, inout);
  }

  std::vector<f_t> out(m, 0.0);
  matrix_vector_multiply(L0_, 1.0, inout, 0.0, out);
  inout = out;
}

template <typename i_t, typename f_t>
void basis_update_mpf_t<i_t, f_t>::l_transpose_multiply(std::vector<f_t>& inout) const
{
  const i_t m = L0_.m;
  std::vector<f_t> out(m, 0.0);
  matrix_vector_multiply(L0_transpose_, 1.0, inout, 0.0, out);

  inout = out;

  const f_t zero_tol = 1e-13;
  for (i_t k = 0; k < num_updates_; ++k) {
    const i_t u_col = 2 * k;
    const i_t v_col = 2 * k + 1;
    const f_t mu    = mu_values_[k];

    // T_k = ( I + u v^T)
    // T_k^T = ( I + v u^T)
    // T_k^T * b = b + v * (u^T * b) = b + theta * v, theta = u^T * b
    f_t dot         = dot_product(u_col, inout);
    const f_t theta = dot;
    if (std::abs(theta) > zero_tol) { add_sparse_column(S_, v_col, theta, inout); }
  }
}

template <typename i_t, typename f_t>
void basis_update_mpf_t<i_t, f_t>::multiply_lu(csc_matrix_t<i_t, f_t>& out) const
{
  // P*B = L*U
  // B = P'*L*U
  const i_t m = L0_.m;

  out.col_start.resize(m + 1);
  out.col_start[0] = 0;
  out.i.clear();
  out.x.clear();

  i_t B_nz = 0;

  for (i_t j = 0; j < m; ++j) {
    // B(:, j) = L*U(:, j)
    out.col_start[j] = B_nz;

    std::vector<f_t> Uj(m, 0.0);
    U0_.load_a_column(j, Uj);
    l_multiply(Uj);
    for (i_t i = 0; i < m; ++i) {
      if (Uj[i] != 0.0) {
        out.i.push_back(row_permutation_[i]);
        out.x.push_back(Uj[i]);
        B_nz++;
      }
    }
  }
  out.col_start[m] = B_nz;

  out.m      = m;
  out.n      = m;
  out.nz_max = B_nz;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class basis_update_t<int, double>;
template class basis_update_mpf_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::dual_simplex
