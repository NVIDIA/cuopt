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

#include <dual_simplex/sparse_matrix.hpp>

#include <dual_simplex/types.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
void csc_matrix_t<i_t, f_t>::reallocate(i_t new_nz)
{
  this->i.resize(new_nz);
  this->x.resize(new_nz);
  this->nz_max = new_nz;
}

template <typename i_t>
void cumulative_sum(std::vector<i_t>& inout, std::vector<i_t>& output)
{
  i_t n = inout.size();
  assert(output.size() == n + 1);
  i_t nz = 0;
  for (i_t i = 0; i < n; ++i) {
    output[i] = nz;
    nz += inout[i];
    inout[i] = output[i];
  }
  output[n] = nz;
}

template <typename i_t, typename f_t>
i_t coo_to_csc(const std::vector<i_t>& Ai,
               const std::vector<i_t>& Aj,
               const std::vector<f_t>& Ax,
               csc_matrix_t<i_t, f_t>& A)
{
  assert(Ai.size() == Aj.size() && Aj.size() == Ax.size());
  if (A.nz_max < Ai.size()) { A.reallocate(static_cast<i_t>(Ai.size())); }

  i_t n  = A.n;
  i_t m  = A.m;
  i_t nz = Aj.size();
  std::vector<i_t> workspace(n);

  // Get the column counts
  for (i_t k = 0; k < nz; ++k) {
    workspace[Aj[k]]++;
  }

  cumulative_sum(workspace, A.col_start);
  for (i_t k = 0; k < nz; ++k) {
    i_t p  = workspace[Aj[k]]++;
    A.i[p] = Ai[k];
    A.x[p] = Ax[k];
  }

  assert(A.col_start[n] == nz);
  return 0;
}

template <typename i_t, typename f_t>
i_t csc_matrix_t<i_t, f_t>::to_compressed_row(csr_matrix_t<i_t, f_t>& Arow) const
{
  i_t m = Arow.m = this->m;
  i_t n = Arow.n = this->n;
  i_t nz         = this->col_start[n];
  Arow.row_start.resize(m + 1);
  Arow.j.resize(nz);
  Arow.x.resize(nz);

  std::vector<i_t> workspace(m, 0);
  for (i_t p = 0; p < nz; ++p) {
    workspace[this->i[p]]++;
  }
  cumulative_sum(workspace, Arow.row_start);
  for (i_t j = 0; j < n; ++j) {
    i_t col_start = this->col_start[j];
    i_t col_end   = this->col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      i_t q     = workspace[this->i[p]]++;
      Arow.j[q] = j;
      Arow.x[q] = this->x[p];
    }
  }
  assert(Arow.row_start[m] == nz);
  return 0;
}

template <typename i_t, typename f_t>
i_t csr_matrix_t<i_t, f_t>::to_compressed_col(csc_matrix_t<i_t, f_t>& Acol) const
{
  i_t m = Acol.m = this->m;
  i_t n = Acol.n = this->n;
  i_t nz         = this->row_start[m];
  Acol.col_start.resize(n + 1);
  Acol.i.resize(nz);
  Acol.x.resize(nz);

  std::vector<i_t> workspace(n, 0);
  for (i_t p = 0; p < nz; ++p) {
    workspace[this->j[p]]++;
  }
  cumulative_sum(workspace, Acol.col_start);
  for (i_t i = 0; i < m; ++i) {
    i_t row_start = this->row_start[i];
    i_t row_end   = this->row_start[i + 1];
    for (i_t p = row_start; p < row_end; ++p) {
      i_t q     = workspace[this->j[p]]++;
      Acol.i[q] = i;
      Acol.x[q] = this->x[p];
    }
  }
  assert(Acol.col_start[n] == nz);
  return 0;
}

template <typename i_t, typename f_t>
i_t csc_matrix_t<i_t, f_t>::load_a_column(i_t j, std::vector<f_t>& Aj) const
{
  assert(Aj.size() == this->m);
  const i_t col_start = this->col_start[j];
  const i_t col_end   = this->col_start[j + 1];
  for (i_t p = col_start; p < col_end; ++p) {
    const i_t i = this->i[p];
    const f_t x = this->x[p];
    Aj[i]       = x;
  }
  return (col_end - col_start);
}

template <typename i_t, typename f_t>
i_t csc_matrix_t<i_t, f_t>::transpose(csc_matrix_t<i_t, f_t>& AT) const
{
  AT.m   = this->n;
  AT.n   = this->m;
  i_t nz = this->col_start[this->n];
  AT.col_start.resize(AT.n + 1);
  AT.reallocate(nz);

  std::vector<i_t> workspace(AT.n, 0);
  for (i_t p = 0; p < nz; ++p) {
    workspace[this->i[p]]++;
  }
  cumulative_sum(workspace, AT.col_start);
  for (i_t j = 0; j < this->n; j++) {
    i_t col_start = this->col_start[j];
    i_t col_end   = this->col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      i_t q   = workspace[this->i[p]]++;
      AT.i[q] = j;
      AT.x[q] = this->x[p];
    }
  }
  assert(AT.col_start[AT.n] == nz);
  return 0;
}

template <typename i_t, typename f_t>
i_t csc_matrix_t<i_t, f_t>::remove_columns(const std::vector<i_t>& cols_to_remove)
{
  // Get a count of new nz
  i_t nz = 0;
  i_t n  = 0;
  for (i_t j = 0; j < this->n; ++j) {
    if (!cols_to_remove[j]) {
      n++;
      nz += this->col_start[j + 1] - this->col_start[j];
    }
  }
  std::vector<i_t> new_i(nz);
  std::vector<f_t> new_x(nz);
  std::vector<i_t> new_col_start(n + 1);

  i_t q     = 0;
  i_t new_j = 0;
  for (i_t j = 0; j < this->n; ++j) {
    if (cols_to_remove[j]) { continue; }
    new_col_start[new_j] = q;
    for (i_t p = this->col_start[j]; p < this->col_start[j + 1]; ++p) {
      new_i[q] = this->i[p];
      new_x[q] = this->x[p];
      q++;
    }
    new_j++;
  }
  new_col_start[n] = q;  // Finalize last column

  this->col_start = new_col_start;
  this->i         = new_i;
  this->x         = new_x;
  this->n         = n;

  return 0;
}

template <typename i_t, typename f_t>
i_t csc_matrix_t<i_t, f_t>::remove_column(i_t col)
{
  const i_t n = this->n;

  i_t q = this->col_start[col];
  for (i_t k = col + 1; k < n; ++k) {
    this->col_start[k - 1] = q;
    const i_t col_start    = this->col_start[k];
    const i_t col_end      = this->col_start[k + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      this->i[q] = this->i[p];
      this->x[q] = this->x[p];
      q++;
    }
  }
  this->col_start[n - 1] = q;  // Finalize A[n-1]

  return 0;
}

template <typename i_t, typename f_t>
i_t csr_matrix_t<i_t, f_t>::remove_rows(std::vector<i_t>& row_marker,
                                        csr_matrix_t<i_t, f_t>& Aout) const
{
  Aout.n                 = this->n;
  i_t num_rows_to_remove = 0;
  i_t num_nnz_to_remove  = 0;
  for (i_t i = 0; i < this->m; i++) {
    if (row_marker[i]) {
      num_rows_to_remove++;
      num_nnz_to_remove += (this->row_start[i + 1] - this->row_start[i]);
    }
  }
  i_t new_rows = this->m - num_rows_to_remove;
  Aout.m       = new_rows;
  i_t new_nnz  = this->row_start[this->m] - num_nnz_to_remove;
  Aout.row_start.resize(new_rows + 1);
  Aout.j.resize(new_nnz);
  Aout.x.resize(new_nnz);

  i_t row_count = 0;
  i_t nz_count  = 0;
  for (i_t i = 0; i < this->m; i++) {
    if (!row_marker[i]) {
      Aout.row_start[row_count] = nz_count;
      for (i_t p = this->row_start[i]; p < this->row_start[i + 1]; p++) {
        Aout.j[nz_count] = this->j[p];
        Aout.x[nz_count] = this->x[p];
        nz_count++;
      }
      row_count++;
    }
  }
  Aout.row_start[new_rows] = new_nnz;
  return 0;
}

template <typename i_t, typename f_t>
i_t csc_matrix_t<i_t, f_t>::remove_row(i_t row)
{
  const i_t n = this->n;
  i_t q       = 0;
  std::vector<i_t> new_col_start(n + 1);
  for (i_t j = 0; j < n; ++j) {
    new_col_start[j]    = q;
    const i_t col_start = this->col_start[j];
    const i_t col_end   = this->col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      i_t i = this->i[p];
      if (i == row) {
        // do nothing
      } else if (i > row) {
        this->i[q] = i - 1;
        this->x[q] = this->x[p];
        q++;
      } else {
        this->i[q] = i;
        this->x[q] = this->x[p];
        q++;
      }
    }
  }
  new_col_start[n] = q;  // Finalize A
  this->col_start  = new_col_start;

  return 0;
}

template <typename i_t, typename f_t>
void csc_matrix_t<i_t, f_t>::print_matrix(FILE* fid) const
{
  fprintf(fid, "ijx = [\n");
  for (i_t j = 0; j < this->n; ++j) {
    i_t p2 = this->col_start[j + 1];
    for (i_t p = this->col_start[j]; p < p2; ++p) {
      fprintf(fid, "%d %d %.16e;\n", this->i[p] + 1, j + 1, this->x[p]);
    }
  }
  fprintf(fid, "];\n");
  fprintf(fid, "A = sparse(ijx(:, 1), ijx(:, 2), ijx(:, 3), %d, %d);\n", this->m, this->n);
}

template <typename i_t, typename f_t>
void csc_matrix_t<i_t, f_t>::print_matrix() const
{
  this->print_matrix(stdout);
}

template <typename i_t, typename f_t>
i_t scatter(const csc_matrix_t<i_t, f_t>& A,
            i_t j,
            f_t beta,
            std::vector<i_t>& workspace,
            std::vector<f_t>& x,
            i_t mark,
            csc_matrix_t<i_t, f_t>& C,
            i_t nz)
{
  i_t col_start = A.col_start[j];
  i_t col_end   = A.col_start[j + 1];
  for (i_t p = col_start; p < col_end; ++p) {
    i_t i = A.i[p];  // A(i, j) is nonzero
    if (workspace[i] < mark) {
      workspace[i] = mark;           // i is a new entry in column j
      C.i[nz++]    = i;              // Add i to the pattern of C(:, j)
      x[i]         = beta * A.x[p];  // x(i) = beta*A(i, j)
    } else {
      x[i] += beta * A.x[p];  // i exists in C(:, j) already
    }
  }
  return nz;
}

// x <- x + alpha * A(:, j)
template <typename i_t, typename f_t>
void scatter_dense(const csc_matrix_t<i_t, f_t>& A, i_t j, f_t alpha, std::vector<f_t>& x)
{
  const i_t col_start = A.col_start[j];
  const i_t col_end   = A.col_start[j + 1];
  for (i_t p = col_start; p < col_end; ++p) {
    const i_t i  = A.i[p];
    const f_t ax = A.x[p];
    x[i] += alpha * ax;
  }
}

// Compute C = A*B where C is m x n, A is m x k, and B = k x n
// Do this by computing C(:, j) = A*B(:, j) = sum (i=1 to k) A(:, k)*B(i, j)
template <typename i_t, typename f_t>
i_t multiply(const csc_matrix_t<i_t, f_t>& A,
             const csc_matrix_t<i_t, f_t>& B,
             csc_matrix_t<i_t, f_t>& C)
{
  i_t m = A.m;
  i_t n = B.n;
  assert(A.n == B.m);
  std::vector<i_t> workspace(m, 0);
  std::vector<f_t> x(m, 0.0);

  C.reallocate(A.nz_max + B.nz_max);

  i_t nz = 0;
  for (int j = 0; j < n; ++j) {
    // Grow C if we don't have enough space
    if (nz + m > C.nz_max) { C.reallocate(2 * C.nz_max + m); }

    C.col_start[j] = nz;  // Column j of C starts here
    i_t col_end    = B.col_start[j + 1];
    for (i_t p = B.col_start[j]; p < col_end; ++p) {
      nz = scatter(A, B.i[p], B.x[p], workspace, x, j + 1, C, nz);
    }

    for (i_t p = C.col_start[j]; p < nz; ++p) {
      C.x[p] = x[C.i[p]];
    }
  }
  C.col_start[n] = nz;  // Finalize the last column of C

  // Remove extra space from C
  C.reallocate(nz);
  return 0;
}

// Compute C = alpha*A + beta*B
template <typename i_t, typename f_t>
i_t add(const csc_matrix_t<i_t, f_t>& A,
        const csc_matrix_t<i_t, f_t>& B,
        f_t alpha,
        f_t beta,
        csc_matrix_t<i_t, f_t>& C)
{
  const i_t m = A.m;
  assert(B.m == m);
  assert(C.m == m);
  const i_t n = B.n;
  assert(A.n == n);
  assert(C.n == n);
  std::vector<i_t> workspace(m, 0);
  std::vector<f_t> x(m, 0.0);
  const i_t Anz = A.col_start[n];
  const i_t Bnz = B.col_start[n];
  C.reallocate(Anz + Bnz);

  i_t nz = 0;  // number of nonzeros in C
  for (i_t j = 0; j < n; ++j) {
    C.col_start[j] = nz;
    nz             = scatter(A, j, alpha, workspace, x, j + 1, C, nz);  // alpha * A(:, j)
    nz             = scatter(B, j, beta, workspace, x, j + 1, C, nz);   // beta * B(:, j)
    for (i_t p = C.col_start[j]; p < nz; ++p) {
      C.x[p] = x[C.i[p]];
    }
  }
  C.col_start[n] = nz;  // Finalize the last column of C
  // Remove extra space from C
  C.reallocate(nz);

  return 0;
}

// Permutes rows of a sparse matrix A
// Computes C = A(p, :)
template <typename i_t, typename f_t>
i_t csc_matrix_t<i_t, f_t>::permute_rows(const std::vector<i_t>& pinv,
                                         csc_matrix_t<i_t, f_t>& C) const
{
  i_t m = this->m;
  i_t n = this->n;
  assert(C.m == m);
  assert(C.n == n);

  i_t Anz = this->col_start[n];
  C.reallocate(Anz);
  i_t nz = 0;
  for (i_t j = 0; j < n; ++j) {
    C.col_start[j]    = nz;
    const i_t col_end = this->col_start[j + 1];
    for (i_t p = this->col_start[j]; p < col_end; ++p) {
      C.x[nz] = this->x[p];  // row i of A is row pinv[i] of C
      C.i[nz] = pinv[this->i[p]];
      nz++;
    }
  }
  C.col_start[n] = nz;  // Finalize the last column of C
  return 0;
}

// Permutes the rows and columns of a sparse matrix A
// Compute C = A(p, q)
template <typename i_t, typename f_t>
i_t csc_matrix_t<i_t, f_t>::permute_rows_and_cols(const std::vector<i_t>& pinv,
                                                  const std::vector<i_t>& q,
                                                  csc_matrix_t<i_t, f_t>& C) const
{
  i_t m = this->m;
  i_t n = this->n;
  assert(C.m == m);
  assert(C.n == n);

  i_t Anz = this->col_start[n];
  C.reallocate(Anz);
  i_t nz = 0;
  for (i_t k = 0; k < n; ++k) {
    C.col_start[k]      = nz;
    const i_t j         = q[k];
    const i_t col_start = this->col_start[j];
    const i_t col_end   = this->col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      C.x[nz] = this->x[p];
      C.i[nz] = pinv[this->i[p]];
      nz++;
    }
  }
  C.col_start[n] = nz;  // Finalize the last column of C
  return 0;
}

// Compute || A ||_1 = max_j (sum {i = 1 to m} | A(i, j) | )
template <typename i_t, typename f_t>
f_t csc_matrix_t<i_t, f_t>::norm1() const
{
  i_t n    = this->n;
  f_t norm = 0;
  for (i_t j = 0; j < n; ++j) {
    f_t sum     = 0;
    i_t col_end = this->col_start[j + 1];
    for (i_t p = this->col_start[j]; p < col_end; ++p) {
      sum += std::abs(this->x[p]);
    }
    norm = std::max(sum, norm);
  }
  return norm;
}

template <typename i_t, typename f_t>
f_t vector_norm_inf(const std::vector<f_t>& x)
{
  i_t n = x.size();
  f_t a = 0.0;
  for (i_t j = 0; j < n; ++j) {
    f_t t = std::abs(x[j]);
    if (t > a) { a = t; }
  }
  return a;
}

template <typename i_t, typename f_t>
f_t vector_norm2_squared(const std::vector<f_t>& x)
{
  i_t n   = x.size();
  f_t sum = 0.0;
  for (i_t j = 0; j < n; ++j) {
    sum += x[j] * x[j];
  }
  return sum;
}

template <typename i_t, typename f_t>
f_t vector_norm2(const std::vector<f_t>& x)
{
  return std::sqrt(vector_norm2_squared<i_t, f_t>(x));
}

template <typename i_t, typename f_t>
f_t dot(const std::vector<f_t>& x, const std::vector<f_t>& y)
{
  assert(x.size() == y.size());
  const i_t n = x.size();
  f_t dot     = 0.0;
  for (i_t k = 0; k < n; ++k) {
    dot += x[k] * y[k];
  }
  return dot;
}

template <typename i_t, typename f_t>
f_t sparse_dot(const std::vector<i_t>& xind,
               const std::vector<f_t>& xval,
               const std::vector<i_t>& yind,
               const std::vector<f_t>& yval)
{
  const i_t nx = xind.size();
  const i_t ny = yind.size();
  f_t dot      = 0.0;
  for (i_t i = 0, j = 0; i < nx && j < ny;) {
    const i_t p = xind[i];
    const i_t q = yind[j];
    if (p == q) {
      dot += xval[i] * yval[j];
      i++;
      j++;
    } else if (p < q) {
      i++;
    } else if (q < p) {
      j++;
    }
  }
  return dot;
}

template <typename i_t, typename f_t>
f_t sparse_dot(const std::vector<i_t>& xind,
               const std::vector<f_t>& xval,
               const csc_matrix_t<i_t, f_t>& Y,
               i_t y_col)
{
  const i_t nx        = xind.size();
  const i_t col_start = Y.col_start[y_col];
  const i_t col_end   = Y.col_start[y_col + 1];
  const i_t ny        = col_end - col_start;
  f_t dot             = 0.0;
  for (i_t i = 0, k = col_start; i < nx && k < col_end;) {
    const i_t p = xind[i];
    const i_t q = Y.i[k];
    if (p == q) {
      dot += Y.x[k] * xval[i];
      i++;
      k++;
    } else if (p < q) {
      i++;
    } else if (q < p) {
      k++;
    }
  }
  return dot;
}

template <typename i_t, typename f_t>
i_t matrix_vector_multiply(const csc_matrix_t<i_t, f_t>& A,
                           f_t alpha,
                           const std::vector<f_t>& x,
                           f_t beta,
                           std::vector<f_t>& y)
{
  // y <- alpha*A*x + beta*y
  i_t m = A.m;
  i_t n = A.n;
  assert(y.size() == m);
  assert(x.size() == n);

  // y <- alpha * sum_j A(:, j)*x_j + beta * y

  // y <- beta * y
  if (beta != 1.0) {
    for (i_t i = 0; i < m; ++i) {
      y[i] *= beta;
    }
  }

  // y <- alpha * sum_j A(:, j)*x_j + y
  for (i_t j = 0; j < n; ++j) {
    i_t col_start = A.col_start[j];
    i_t col_end   = A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      i_t i = A.i[p];
      y[i] += alpha * A.x[p] * x[j];
    }
  }

  return 0;
}

// y <- alpha*A'*x + beta*y
template <typename i_t, typename f_t>
i_t matrix_transpose_vector_multiply(const csc_matrix_t<i_t, f_t>& A,
                                     f_t alpha,
                                     const std::vector<f_t>& x,
                                     f_t beta,
                                     std::vector<f_t>& y)
{
  i_t m = A.m;
  i_t n = A.n;
  assert(y.size() == n);
  assert(x.size() == m);

  // y <- beta * y
  if (beta != 1.0) {
    for (i_t j = 0; j < n; ++j) {
      y[j] *= beta;
    }
  }

  // y <- alpha * AT*x + y
  for (i_t j = 0; j < n; ++j) {
    f_t dot       = 0.0;
    i_t col_start = A.col_start[j];
    i_t col_end   = A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      dot += A.x[p] * x[A.i[p]];
    }
    y[j] += alpha * dot;
  }

  return 0;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class csc_matrix_t<int, double>;

template class csr_matrix_t<int, double>;

template void cumulative_sum<int>(std::vector<int>& inout, std::vector<int>& output);

template int coo_to_csc<int, double>(const std::vector<int>& Ai,
                                     const std::vector<int>& Aj,
                                     const std::vector<double>& Ax,
                                     csc_matrix_t<int, double>& A);

template int scatter<int, double>(const csc_matrix_t<int, double>& A,
                                  int j,
                                  double beta,
                                  std::vector<int>& workspace,
                                  std::vector<double>& x,
                                  int mark,
                                  csc_matrix_t<int, double>& C,
                                  int nz);

template void scatter_dense<int, double>(const csc_matrix_t<int, double>& A,
                                         int j,
                                         double alpha,
                                         std::vector<double>& x);

template int multiply<int, double>(const csc_matrix_t<int, double>& A,
                                   const csc_matrix_t<int, double>& B,
                                   csc_matrix_t<int, double>& C);

template int add<int, double>(const csc_matrix_t<int, double>& A,
                              const csc_matrix_t<int, double>& B,
                              double alpha,
                              double beta,
                              csc_matrix_t<int, double>& C);

template double sparse_dot<int, double>(const std::vector<int>& xind,
                                        const std::vector<double>& xval,
                                        const csc_matrix_t<int, double>& Y,
                                        int y_col);

template int matrix_vector_multiply<int, double>(const csc_matrix_t<int, double>& A,
                                                 double alpha,
                                                 const std::vector<double>& x,
                                                 double beta,
                                                 std::vector<double>& y);

template int matrix_transpose_vector_multiply<int, double>(const csc_matrix_t<int, double>& A,
                                                           double alpha,
                                                           const std::vector<double>& x,
                                                           double beta,
                                                           std::vector<double>& y);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
