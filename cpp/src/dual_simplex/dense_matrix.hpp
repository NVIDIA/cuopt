/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <dual_simplex/dense_vector.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>

#pragma once

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class dense_matrix_t {
 public:
  dense_matrix_t(i_t rows, i_t cols) : m(rows), n(cols), values(rows * cols, 0.0) {}

  void resize(i_t rows, i_t cols)
  {
    m = rows;
    n = cols;
    values.resize(rows * cols, 0.0);
  }

  f_t& operator()(i_t row, i_t col) { return values[col * m + row]; }

  f_t operator()(i_t row, i_t col) const { return values[col * m + row]; }

  void from_sparse(const csc_matrix_t<i_t, f_t>& A, i_t sparse_column, i_t dense_column)
  {
    for (i_t i = 0; i < m; i++) {
      this->operator()(i, dense_column) = 0.0;
    }

    const i_t col_start = A.col_start[sparse_column];
    const i_t col_end   = A.col_start[sparse_column + 1];
    for (i_t p = col_start; p < col_end; p++) {
      this->operator()(A.i[p], dense_column) = A.x[p];
    }
  }

  void add_diagonal(const dense_vector_t<i_t, f_t>& diag)
  {
    for (i_t j = 0; j < n; j++) {
      this->operator()(j, j) += diag[j];
    }
  }

  void set_column(i_t col, const dense_vector_t<i_t, f_t>& x)
  {
    for (i_t i = 0; i < m; i++) {
      this->operator()(i, col) = x[i];
    }
  }

  // y = alpha * A * x + beta * y
  void matrix_vector_multiply(f_t alpha,
                              const dense_vector_t<i_t, f_t>& x,
                              f_t beta,
                              dense_vector_t<i_t, f_t>& y) const
  {
    if (static_cast<i_t>(x.size()) != n) {
      printf("dense_matrix_t::matrix_vector_multiply: x.size() != n\n");
      exit(1);
    }
    if (static_cast<i_t>(y.size()) != m) {
      printf("dense_matrix_t::matrix_vector_multiply: y.size() != m\n");
      exit(1);
    }

    for (i_t i = 0; i < m; i++) {
      y[i] *= beta;
    }

    const dense_matrix_t<i_t, f_t>& A = *this;

    for (i_t j = 0; j < n; j++) {
      for (i_t i = 0; i < m; i++) {
        y[i] += alpha * A(i, j) * x[j];
      }
    }
  }

  // y = alpha * A' * x + beta * y
  void transpose_multiply(f_t alpha,
                          const dense_vector_t<i_t, f_t>& x,
                          f_t beta,
                          dense_vector_t<i_t, f_t>& y) const
  {
    if (static_cast<i_t>(x.size()) != m) {
      printf("dense_matrix_t::transpose_multiply: x.size() != n\n");
      exit(1);
    }
    for (i_t j = 0; j < n; j++) {
      f_t sum = 0.0;
      for (i_t i = 0; i < m; i++) {
        sum += x[i] * this->operator()(i, j);
      }
      y[j] = alpha * sum + beta * y[j];
    }
  }

  void transpose_multiply(f_t alpha, f_t* x, f_t beta, f_t* y) const
  {
    for (i_t j = 0; j < n; j++) {
      f_t sum = 0.0;
      for (i_t i = 0; i < m; i++) {
        sum += x[i] * this->operator()(i, j);
      }
      y[j] = alpha * sum + beta * y[j];
    }
  }

  // Y <- alpha * A' * X + beta * Y
  void transpose_matrix_multiply(f_t alpha,
                                 const dense_matrix_t<i_t, f_t>& X,
                                 f_t beta,
                                 dense_matrix_t<i_t, f_t>& Y) const
  {
    // X is m x p
    // Y is q x p
    // Y <- alpha * A' * X + beta * Y
    if (X.n != Y.n) {
      printf("dense_matrix_t::transpose_matrix_multiply: X.m != Y.m\n");
      exit(1);
    }
    if (X.m != m) {
      printf("dense_matrix_t::transpose_matrix_multiply: X.m != m\n");
      exit(1);
    }
    for (i_t k = 0; k < X.n; k++) {
      for (i_t j = 0; j < n; j++) {
        f_t sum = 0.0;
        for (i_t i = 0; i < m; i++) {
          sum += this->operator()(i, j) * X(i, k);
        }
        Y(j, k) = alpha * sum + beta * Y(j, k);
      }
    }
  }

  void scale_columns(const dense_vector_t<i_t, f_t>& scale)
  {
    for (i_t j = 0; j < n; j++) {
      for (i_t i = 0; i < m; i++) {
        this->operator()(i, j) *= scale[j];
      }
    }
  }

  void chol(dense_matrix_t<i_t, f_t>& L)
  {
    if (m != n) {
      printf("dense_matrix_t::chol: m != n\n");
      exit(1);
    }
    if (L.m != n) {
      printf("dense_matrix_t::chol: L.m != n\n");
      exit(1);
    }

    // Clear the upper triangular part of L
    for (i_t i = 0; i < n; i++) {
      for (i_t j = i + 1; j < n; j++) {
        L(i, j) = 0.0;
      }
    }

    const dense_matrix_t<i_t, f_t>& A = *this;
    // Compute the cholesky factor and store it in the lower triangular part of L
    for (i_t i = 0; i < n; i++) {
      for (i_t j = 0; j <= i; j++) {
        f_t sum = 0.0;
        for (i_t k = 0; k < j; k++) {
          sum += L(i, k) * L(j, k);
        }

        if (i == j) {
          L(i, j) = sqrt(A(i, i) - sum);
        } else {
          L(i, j) = (1.0 / L(j, j) * (A(i, j) - sum));
        }
      }
    }
  }

  // Assume A = L
  // Solve L * x = b
  void triangular_solve(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x)
  {
    if (static_cast<i_t>(b.size()) != n) {
      printf(
        "dense_matrix_t::triangular_solve: b.size() %d != n %d\n", static_cast<i_t>(b.size()), n);
      exit(1);
    }
    x.resize(n, 0.0);

    // sum_k=0^i-1 L(i, k) * x[k] + L(i, i) * x[i] = b[i]
    // x[i] = (b[i] - sum_k=0^i-1 L(i, k) * x[k]) / L(i, i)
    const dense_matrix_t<i_t, f_t>& L = *this;
    for (i_t i = 0; i < n; i++) {
      f_t sum = 0.0;
      for (i_t k = 0; k < i; k++) {
        sum += L(i, k) * x[k];
      }
      x[i] = (b[i] - sum) / L(i, i);
    }
  }

  // Assume A = L
  // Solve  L^T * x = b
  void triangular_solve_transpose(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x)
  {
    if (static_cast<i_t>(b.size()) != n) {
      printf("dense_matrix_t::triangular_solve_transpose: b.size() != n\n");
      exit(1);
    }
    x.resize(n, 0.0);

    // L^T = U
    // U * x = b
    // sum_k=i+1^n U(i, k) * x[k] + U(i, i) * x[i] = b[i], i=n-1, n-2, ..., 0
    // sum_k=i+1^n L(k, i) * x[k] + L(i, i) * x[i] = b[i], i=n-1, n-2, ..., 0
    const dense_matrix_t<i_t, f_t>& L = *this;
    for (i_t i = n - 1; i >= 0; i--) {
      f_t sum = 0.0;
      for (i_t k = i + 1; k < n; k++) {
        sum += L(k, i) * x[k];
      }
      x[i] = (b[i] - sum) / L(i, i);
    }
  }

  i_t m;
  i_t n;
  std::vector<f_t> values;
};

}  // namespace cuopt::linear_programming::dual_simplex
