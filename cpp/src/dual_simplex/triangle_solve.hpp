/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/sparse_vector.hpp>
#include <dual_simplex/types.hpp>

#include <optional>

namespace cuopt::linear_programming::dual_simplex {

#define FLIP(i)      (-(i) - 2)  // flips an unsigned integer about -1
#define UNFLIP(i)    (((i) < 0) ? FLIP(i) : (i))
#define MARKED(w, j) (w[j] < 0)
#define MARK(w, j)     \
  {                    \
    w[j] = FLIP(w[j]); \
  }

// Solve L*x = b. On input x contains the right-hand side b, on output the
// solution
template <typename i_t, typename f_t>
i_t lower_triangular_solve(const csc_matrix_t<i_t, f_t>& L, std::vector<f_t>& x, f_t& work_estimate)
{
  i_t n = L.n;
  assert(x.size() == n);

  for (i_t j = 0; j < n; ++j) {
    i_t col_start = L.col_start[j];
    i_t col_end   = L.col_start[j + 1];
    if (x[j] != 0.0) {
      x[j] /= L.x[col_start];
      const f_t x_j = x[j];  // hoist this load out of the loop
      // as the compiler cannot guess that x[j] never aliases to x[L.i[p]]
      for (i_t p = col_start + 1; p < col_end; ++p) {
        x[L.i[p]] -= L.x[p] * x_j;
      }
    }
  }
  work_estimate += 3 * n + 3 * L.col_start[n];
  return 0;
}

// Solve L'*x = b. On input x contains the right-hand side b, on output the
// solution
template <typename i_t, typename f_t>
i_t lower_triangular_transpose_solve(const csc_matrix_t<i_t, f_t>& L,
                                     std::vector<f_t>& x,
                                     f_t& work_estimate)
{
  const i_t n = L.n;
  assert(x.size() == n);

  for (i_t j = n - 1; j >= 0; --j) {
    const i_t col_start = L.col_start[j] + 1;
    const i_t col_end   = L.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      x[j] -= L.x[p] * x[L.i[p]];
    }
    x[j] /= L.x[L.col_start[j]];
  }
  work_estimate += 6 * n + 4 * L.col_start[n];
  return 0;
}

// Solve U*x = b. On input x contains the right-hand side b, on output the
// solution
template <typename i_t, typename f_t>
i_t upper_triangular_solve(const csc_matrix_t<i_t, f_t>& U, std::vector<f_t>& x, f_t& work_estimate)
{
  const i_t n = U.n;
  assert(x.size() == n);
  for (i_t j = n - 1; j >= 0; --j) {
    const i_t col_start = U.col_start[j];
    const i_t col_end   = U.col_start[j + 1] - 1;
    if (x[j] != 0.0) {
      x[j] /= U.x[col_end];
      const f_t x_j = x[j];  // same x_j load hoisting
      for (i_t p = col_start; p < col_end; ++p) {
        x[U.i[p]] -= U.x[p] * x_j;
      }
      work_estimate += 3 * (col_end - col_start) + 3;
    }
  }
  work_estimate += 3 * n;
  return 0;
}

// Solve U'*x = b. On input x contains the right-hand side b, on output the
// solution
template <typename i_t, typename f_t>
i_t upper_triangular_transpose_solve(const csc_matrix_t<i_t, f_t>& U,
                                     std::vector<f_t>& x,
                                     f_t& work_estimate)
{
  const i_t n = U.n;
  assert(x.size() == n);
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = U.col_start[j];
    const i_t col_end   = U.col_start[j + 1] - 1;
    for (i_t p = col_start; p < col_end; ++p) {
      x[j] -= U.x[p] * x[U.i[p]];
    }
    x[j] /= U.x[col_end];
  }
  work_estimate += 4 * n + 4 * U.col_start[n];
  return 0;
}

// \brief Reach computes the reach of b in the graph of G
// \param[in] b - sparse vector containing the rhs
// \param[in] pinv - inverse permuation vector
// \param[in, out] G - Sparse CSC matrix G. The column pointers of G are
// modified (but restored) during this call \param[out] xi  - stack of size 2*n.
// xi[top] .. xi[n-1] contains the reachable indicies \returns top - the size of
// the stack
template <typename i_t, typename f_t>
i_t reach(const sparse_vector_t<i_t, f_t>& b,
          const std::optional<std::vector<i_t>>& pinv,
          csc_matrix_t<i_t, f_t>& G,
          std::vector<i_t>& xi,
          f_t& work_estimate);

// \brief Performs a depth-first search starting from node j in the graph
// defined by G
// \param[in] j - root node
// \param[in] pinv - inverse permutation
// \param[in, out] G - graph defined by sparse CSC matrix
// \param[in, out] top - top of the stack in xi
// \param[in, out] xi  - stack containing the nodes found in topological order
// \param[in, out] pstack - private stack (points into xi)
//
// \brief A node j is marked by flipping G.col_start[j]. This exploits the fact
// that G.col_start[j] >= 0 in an unmodified matrix.
//        A marked node will have G.col_start[j] < 0. To unmark a node or to
//        obtain the original value of G.col_start[j] we flip it again. Since
//        flip is its own inverse. UNFLIP(i) flips i if i < 0, or returns i
//        otherwise
template <typename i_t, typename f_t>
i_t depth_first_search(i_t j,
                       const std::optional<std::vector<i_t>>& pinv,
                       csc_matrix_t<i_t, f_t>& G,
                       i_t top,
                       std::vector<i_t>& xi,
                       typename std::vector<i_t>::iterator pstack,
                       f_t& work_estimate);

// \brief sparse_triangle_solve solve L*x = b or U*x = b where L is a sparse lower
// triangular matrix
//        and U is a sparse upper triangular matrix, and b is a sparse
//        right-hand side. The vector b is obtained from the column of a sparse
//        matrix.
// \param[in] b - Sparse vector contain the rhs
// \param[in] pinv - optional inverse permutation vector
// \param[in, out] xi - An array of size 2*m, on output it contains the non-zero
// pattern of x in xi[top] through xi[m-1]
// \param[in, out] G - The lower triangular matrix L or the upper triangular matrix U
//                     G.col_start is marked and restored during the algorithm
// \param[out] - The solution vector xi_t
template <typename i_t, typename f_t, bool lo>
i_t sparse_triangle_solve(const sparse_vector_t<i_t, f_t>& b,
                          const std::optional<std::vector<i_t>>& pinv,
                          std::vector<i_t>& xi,
                          csc_matrix_t<i_t, f_t>& G,
                          f_t* x,
                          f_t& work_estimate);

}  // namespace cuopt::linear_programming::dual_simplex
