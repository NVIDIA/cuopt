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

#include <dual_simplex/types.hpp>
#include <dual_simplex/vector_math.hpp>

#include <cassert>
#include <cstdio>
#include <vector>
#include <algorithm>

namespace cuopt::linear_programming::dual_simplex {


template <typename i_t, typename f_t>
class csr_matrix_t;  // Forward declaration of CSR matrix needed to define CSC matrix

// A sparse matrix stored in compressed sparse column format
template <typename i_t, typename f_t>
class csc_matrix_t {
 public:
  csc_matrix_t(i_t rows, i_t cols, i_t nz)
    : m(rows), n(cols), nz_max(nz), col_start(n + 1), i(nz_max), x(nz_max)
  {
  }

  // Adjust to i and x vectors for a new number of nonzeros
  void reallocate(i_t new_nz);

  // Convert the CSC matrix to a CSR matrix
  i_t to_compressed_row(
    cuopt::linear_programming::dual_simplex::csr_matrix_t<i_t, f_t>& Arow) const;

  // Permutes rows of a sparse matrix A. Computes C = A(p, :)
  i_t permute_rows(const std::vector<i_t>& pinv, csc_matrix_t<i_t, f_t>& C) const;

  // Permutes rows and columns of a sparse matrix A. Computes C = A(p, q)
  i_t permute_rows_and_cols(const std::vector<i_t>& pinv,
                            const std::vector<i_t>& q,
                            csc_matrix_t<i_t, f_t>& C) const;

  // Aj <- A(:, j), where Aj is a dense vector initially all zero
  i_t load_a_column(i_t j, std::vector<f_t>& Aj) const;

  // Compute the transpose of A
  i_t transpose(csc_matrix_t<i_t, f_t>& AT) const;

  // Remove columns from the matrix
  i_t remove_columns(const std::vector<i_t>& cols_to_remove);

  // Removes a single column from the matrix
  i_t remove_column(i_t col);

  // Removes a single row from the matrix
  i_t remove_row(i_t row);

  // Prints the matrix to stdout
  void print_matrix() const;

  // Prints the matrix to a file
  void print_matrix(FILE* fid) const;

  // Compute || A ||_1 = max_j (sum {i = 1 to m} | A(i, j) | )
  f_t norm1() const;

  i_t nz_max;                  // maximum number of entries
  i_t m;                       // number of rows
  i_t n;                       // number of columns
  std::vector<i_t> col_start;  // column pointers (size n + 1)
  std::vector<i_t> i;          // row indices, size nz_max
  std::vector<f_t> x;          // numerical values, size nz_max

  static_assert(std::is_signed_v<i_t>);  // Require signed integers (we make use of this
                                         // to avoid extra space / computation)
};

// A sparse matrix stored in compressed sparse row format
template <typename i_t, typename f_t>
class csr_matrix_t {
 public:
  // Convert the CSR matrix to CSC
  i_t to_compressed_col(csc_matrix_t<i_t, f_t>& Acol) const;

  // Create a new matrix with the marked rows removed
  i_t remove_rows(std::vector<i_t>& row_marker, csr_matrix_t<i_t, f_t>& Aout) const;

  i_t nz_max;                  // maximum number of nonzero entries
  i_t m;                       // number of rows
  i_t n;                       // number of cols
  std::vector<i_t> row_start;  // row pointers (size m + 1)
  std::vector<i_t> j;          // column inidices, size nz_max
  std::vector<f_t> x;          // numerical valuse, size nz_max

  static_assert(std::is_signed_v<i_t>);
};


template <typename i_t, typename f_t>
class sparse_vector_t {
 public:
  sparse_vector_t(i_t n, i_t nz) : n(n), i(nz), x(nz) {}
  sparse_vector_t(const std::vector<f_t>& in) : n(in.size())
  {
    i_t nz = 0;
    for (i_t k = 0; k < n; ++k) {
      if (in[k] != 0) {
        i.push_back(k);
        x.push_back(in[k]);
      }
    }
  }
  sparse_vector_t(const csc_matrix_t<i_t, f_t>& A, i_t col)
  {
    const i_t col_start = A.col_start[col];
    const i_t col_end = A.col_start[col + 1];
    n = A.n;
    const i_t nz = col_end - col_start;
    i.reserve(nz);
    x.reserve(nz);
    for (i_t k = col_start; k < col_end; ++k) {
      i.push_back(A.i[k]);
      x.push_back(A.x[k]);
    }
  }

  void to_csc(csc_matrix_t<i_t, f_t>& A) const
  {
    A.m = n;
    A.n = 1;
    A.nz_max = i.size();
    A.col_start.clear();
    A.col_start.resize(2);
    A.col_start[0] = 0;
    A.col_start[1] = i.size();
    A.i = i;
    A.x = x;
  }

  void to_dense(std::vector<f_t>& x_dense) const
  {
    x_dense.clear();
    x_dense.resize(n, 0.0);
    const i_t nz = i.size();
    for (i_t k = 0; k < nz; ++k) {
      x_dense[i[k]] = x[k];
    }
  }

  void scatter(std::vector<f_t>& x_dense) const
  {
    // Assumes x_dense is already cleared
    const i_t nz = i.size();
    for (i_t k = 0; k < nz; ++k) {
      x_dense[i[k]] += x[k];
    }
  }

  void inverse_permute_vector(const std::vector<i_t>& p)
  {
    assert(p.size() == n);
    i_t nz = i.size();
    std::vector<i_t> i_perm(nz);
    for (i_t k = 0; k < nz; ++k) {
      i_perm[k] = p[i[k]];
    }
    i = i_perm;
  }

  void inverse_permute_vector(const std::vector<i_t>& p, sparse_vector_t<i_t, f_t>& y) const
  {
    i_t m = p.size();
    assert(n == m);
    i_t nz = i.size();
    y.n = n;
    y.x = x;
    std::vector<i_t> i_perm(nz);
    for (i_t k = 0; k < nz; ++k) {
      i_perm[k] = p[i[k]];
    }
    y.i = i_perm;
  }

  f_t sparse_dot(const csc_matrix_t<i_t, f_t>& Y, i_t y_col) const
  {
    const i_t col_start = Y.col_start[y_col];
    const i_t col_end = Y.col_start[y_col + 1];
    const i_t ny = col_end - col_start;
    const i_t nx = i.size();
    f_t dot = 0.0;
    for (i_t h = 0, k = col_start; h < nx && k < col_end; ) {
      const i_t p = i[h];
      const i_t q = Y.i[k];
      if (p == q) {
        dot += Y.x[k] * x[h];
        h++;
        k++;
      } else if (p < q) {
        h++;
      } else if (q < p) {
        k++;
      }
    }
    return dot;
  }

  void sort()
  {
    if (i.size() < 2) {
      return;
    }
    // If the number of nonzeros is large, use a O(n) bucket sort
    if (i.size() > 0.3 *n)
    {
      std::vector<f_t> bucket(n, 0.0);
      const i_t nz = i.size();
      for (i_t k = 0; k < nz; ++k)
      {
        bucket[i[k]] = x[k];
      }
      i.clear();
      i.reserve(nz);
      x.clear();
      x.reserve(nz);
      for (i_t k = 0; k < n; ++k)
      {
        if (bucket[k] != 0.0)
        {
          i.push_back(k);
          x.push_back(bucket[k]);
        }
      }
    }
    else
    {
      // Use a n log n sort
      const i_t nz = i.size();
      std::vector<i_t> i_sorted(nz);
      std::vector<f_t> x_sorted(nz);
      std::vector<i_t> perm(nz);
      for (i_t k = 0; k < nz; ++k)
      {
        perm[k] = k;
      }
      std::vector<i_t>& iunsorted = i;
      std::sort(perm.begin(), perm.end(), [&iunsorted](i_t a, i_t b) { return iunsorted[a] < iunsorted[b]; });
      for (i_t k = 0; k < nz; ++k)
      {
        i_sorted[k] = i[perm[k]];
        x_sorted[k] = x[perm[k]];
      }
      i = i_sorted;
      x = x_sorted;
    }

    // Check
#ifdef CHECK_SORT
  for (i_t k = 0; k < i.size() - 1; ++k) {
    if (i[k] > i[k + 1]) {
      printf("Sort error %d %d\n", i[k], i[k + 1]);
    }
  }
#endif
  }

  f_t norm2_squared() const
  {
    f_t dot = 0.0;
    const i_t nz = i.size();
    for (i_t k = 0; k < nz; ++k) {
      dot += x[k] * x[k];
    }
    return dot;
  }

  i_t n;
  std::vector<i_t> i;
  std::vector<f_t> x;
};

template <typename i_t>
void cumulative_sum(std::vector<i_t>& inout, std::vector<i_t>& output);

template <typename i_t, typename f_t>
i_t coo_to_csc(const std::vector<i_t>& Ai,
               const std::vector<i_t>& Aj,
               const std::vector<f_t>& Ax,
               csc_matrix_t<i_t, f_t>& A);

template <typename i_t, typename f_t>
i_t scatter(const csc_matrix_t<i_t, f_t>& A,
            i_t j,
            f_t beta,
            std::vector<i_t>& workspace,
            std::vector<f_t>& x,
            i_t mark,
            csc_matrix_t<i_t, f_t>& C,
            i_t nz);

// x <- x + alpha * A(:, j)
template <typename i_t, typename f_t>
void scatter_dense(const csc_matrix_t<i_t, f_t>& A, i_t j, f_t alpha, std::vector<f_t>& x);

// Compute C = A*B where C is m x n, A is m x k, and B = k x n
// Do this by computing C(:, j) = A*B(:, j) = sum (i=1 to k) A(:, k)*B(i, j)
template <typename i_t, typename f_t>
i_t multiply(const csc_matrix_t<i_t, f_t>& A,
             const csc_matrix_t<i_t, f_t>& B,
             csc_matrix_t<i_t, f_t>& C);

// Compute C = alpha*A + beta*B
template <typename i_t, typename f_t>
i_t add(const csc_matrix_t<i_t, f_t>& A,
        const csc_matrix_t<i_t, f_t>& B,
        f_t alpha,
        f_t beta,
        csc_matrix_t<i_t, f_t>& C);

template <typename i_t, typename f_t>
f_t sparse_dot(const std::vector<i_t>& xind,
               const std::vector<f_t>& xval,
               const csc_matrix_t<i_t, f_t>& Y,
               i_t y_col);

// y <- alpha*A*x + beta*y
template <typename i_t, typename f_t>
i_t matrix_vector_multiply(const csc_matrix_t<i_t, f_t>& A,
                           f_t alpha,
                           const std::vector<f_t>& x,
                           f_t beta,
                           std::vector<f_t>& y);

// y <- alpha*A'*x + beta*y
template <typename i_t, typename f_t>
i_t matrix_transpose_vector_multiply(const csc_matrix_t<i_t, f_t>& A,
                                     f_t alpha,
                                     const std::vector<f_t>& x,
                                     f_t beta,
                                     std::vector<f_t>& y);

}  // namespace cuopt::linear_programming::dual_simplex
