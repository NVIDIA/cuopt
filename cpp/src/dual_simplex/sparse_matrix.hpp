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

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class csr_matrix_t;  // Forward declaration of CSR matrix needed to define CSC matrix

template <typename i_t, typename f_t>
class sparse_vector_t;  // Forward declaration of sparse vector needed to define CSC matrix

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

  // Append a dense column to the matrix. Assumes the matrix has already been resized accordingly
  void append_column(const std::vector<f_t>& x);

  // Append a sparse column to the matrix. Assumes the matrix has already been resized accordingly
  void append_column(const sparse_vector_t<i_t, f_t>& x);

  // Append a sparse column to the matrix. Assumes the matrix has already been resized accordingly
  void append_column(i_t nz, i_t* i, f_t* x);

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

  i_t m;                       // number of rows
  i_t n;                       // number of columns
  i_t nz_max;                  // maximum number of entries
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

template <typename i_t, typename f_t>
void scatter_dense(const csc_matrix_t<i_t, f_t>& A,
                   i_t j,
                   f_t alpha,
                   std::vector<f_t>& x,
                   std::vector<i_t>& mark,
                   std::vector<i_t>& indices);

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
