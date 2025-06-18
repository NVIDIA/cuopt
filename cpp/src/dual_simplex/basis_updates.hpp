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

#pragma once

#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>

namespace cuopt::linear_programming::dual_simplex {

// Forrest-Tomlin update to the LU factorization of a basis matrix B
template <typename i_t, typename f_t>
class basis_update_t {
 public:
  basis_update_t(const csc_matrix_t<i_t, f_t>& Linit,
                 const csc_matrix_t<i_t, f_t>& Uinit,
                 const std::vector<i_t>& p)
    : L0_(Linit),
      U_(Uinit),
      row_permutation_(p),
      inverse_row_permutation_(p.size()),
      S_(Linit.m, 1, 0),
      col_permutation_(Linit.m),
      inverse_col_permutation_(Linit.m),
      xi_workspace_(2*Linit.m, 0),
      x_workspace_(Linit.m, 0.0),
      U_transpose_(1, 1, 1),
      L0_transpose_(1, 1, 1)
  {
    inverse_permutation(row_permutation_, inverse_row_permutation_);
    clear();
    compute_transposes();
  }

  i_t reset(const csc_matrix_t<i_t, f_t>& Linit,
            const csc_matrix_t<i_t, f_t>& Uinit,
            const std::vector<i_t>& p)
  {
    L0_ = Linit;
    U_  = Uinit;
    assert(p.size() == Linit.m);
    row_permutation_ = p;
    inverse_permutation(row_permutation_, inverse_row_permutation_);
    clear();
    compute_transposes();
    return 0;
  }

  // Solves for x such that B*x = b, where B is the basis matrix
  i_t b_solve(const std::vector<f_t>& rhs, std::vector<f_t>& solution) const;

  // Solves for x such that B*x = b, where B is the basis matrix
  i_t b_solve(const sparse_vector_t<i_t, f_t>& rhs, sparse_vector_t<i_t, f_t>& solution) const;

  // Solves for x such that B*x = b, where B is the basis matrix, also returns L*v = P*b
  // This is useful for avoiding an extra solve with the update
  i_t b_solve(const std::vector<f_t>& rhs,
              std::vector<f_t>& solution,
              std::vector<f_t>& Lsol) const;

  // Solves for x such that B*x = b, where B is the basis matrix, also returns L*v = P*b
  // This is useful for avoiding an extra solve with the update
  i_t b_solve(const sparse_vector_t<i_t, f_t>& rhs,
              sparse_vector_t<i_t, f_t>& solution,
              sparse_vector_t<i_t, f_t>& Lsol) const;

  // Solves for y such that B'*y = c, where B is the basis matrix
  i_t b_transpose_solve(const std::vector<f_t>& rhs, std::vector<f_t>& solution) const;

  i_t b_transpose_solve(const sparse_vector_t<i_t, f_t>& rhs, sparse_vector_t<i_t, f_t>& solution) const;

  // Solve for x such that L*x = y
  i_t l_solve(std::vector<f_t>& rhs) const;

  // Solve for x such that L*x = y
  i_t l_solve(sparse_vector_t<i_t, f_t>& rhs) const;

  // Solve for x such that L'*x = y
  i_t l_transpose_solve(std::vector<f_t>& rhs) const;

  // Solve for x such that L'*x = y
  i_t l_transpose_solve(sparse_vector_t<i_t, f_t>& rhs) const;

  // Solve for x such that U*x = y
  i_t u_solve(std::vector<f_t>& rhs) const;

  // Solve for x such that U*x = y
  i_t u_solve(sparse_vector_t<i_t, f_t>& rhs) const;

  // Solve for x such that U'*x = y
  i_t u_transpose_solve(std::vector<f_t>& rhs) const;

  // Solve for x such that U'*x = y
  i_t u_transpose_solve(sparse_vector_t<i_t, f_t>& rhs) const;

  // Replace the column B(:, leaving_index) with the vector abar. Pass in utilde such that L*utilde
  // = abar
  i_t update(std::vector<f_t>& utilde, i_t leaving_index);

  i_t multiply_lu(csc_matrix_t<i_t, f_t>& out);

  i_t num_updates() const { return num_updates_; }

  const std::vector<i_t>& row_permutation() const { return row_permutation_; }

  void compute_transposes()
  {
    L0_.transpose(L0_transpose_);
    U_.transpose(U_transpose_);
  }

 private:
  void clear()
  {
    pivot_indices_.clear();
    pivot_indices_.reserve(L0_.m);
    for (i_t k = 0; k < L0_.m; ++k) {
      col_permutation_[k]         = k;
      inverse_col_permutation_[k] = k;
    }
    S_.col_start[0] = 0;
    S_.col_start[1] = 0;
    S_.i.clear();
    S_.x.clear();
    num_updates_ = 0;
  }
  i_t index_map(i_t leaving) const;
  f_t u_diagonal(i_t j) const;
  i_t place_diagonals();
  f_t update_lower(const std::vector<i_t>& sind, const std::vector<f_t>& sval, i_t leaving);
  i_t update_upper(const std::vector<i_t>& ind, const std::vector<f_t>& baru, i_t t);
  i_t lower_triangular_multiply(const csc_matrix_t<i_t, f_t>& in,
                                i_t in_col,
                                csc_matrix_t<i_t, f_t>& out,
                                i_t out_col) const;

  void solve_to_sparse_vector(i_t top, sparse_vector_t<i_t, f_t>& out) const;
  i_t scatter_into_workspace(const sparse_vector_t<i_t, f_t>& in) const;
  void gather_into_sparse_vector(i_t nz, sparse_vector_t<i_t, f_t>& out) const;

  i_t num_updates_;                   // Number of rank-1 updates to L0
  mutable csc_matrix_t<i_t, f_t> L0_;         // Sparse lower triangular matrix from initial factorization
  mutable csc_matrix_t<i_t, f_t> U_;          // Sparse upper triangular matrix. Is modified by updates
  std::vector<i_t> row_permutation_;  // Row permutation from initial factorization L*U = P*B
  std::vector<i_t> inverse_row_permutation_;  // Inverse row permutation from initial factorization L*U = P*B
  std::vector<i_t> pivot_indices_;    // indicies for rank-1 updates to L
  csc_matrix_t<i_t, f_t> S_;          // stores the pivot elements for rank-1 updates to L
  std::vector<i_t> col_permutation_;  // symmetric permuation q used in U(q, q) represents Q
  std::vector<i_t> inverse_col_permutation_;  // inverse permutation represents Q'
  mutable std::vector<i_t> xi_workspace_;
  mutable std::vector<f_t> x_workspace_;
  mutable csc_matrix_t<i_t, f_t> U_transpose_; // Needed for sparse solves
  mutable csc_matrix_t<i_t, f_t> L0_transpose_; // Needed for sparse solves
};

}  // namespace cuopt::linear_programming::dual_simplex
