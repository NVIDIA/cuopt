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
i_t lower_triangular_solve(const csc_matrix_t<i_t, f_t>& L, std::vector<f_t>& x);

// Solve L'*x = b. On input x contains the right-hand side b, on output the
// solution
template <typename i_t, typename f_t>
i_t lower_triangular_transpose_solve(const csc_matrix_t<i_t, f_t>& L, std::vector<f_t>& x);

// Solve U*x = b. On input x contains the right-hand side b, on output the
// solution
template <typename i_t, typename f_t>
i_t upper_triangular_solve(const csc_matrix_t<i_t, f_t>& U, std::vector<f_t>& x);

// Solve U'*x = b. On input x contains the right-hand side b, on output the
// solution
template <typename i_t, typename f_t>
i_t upper_triangular_transpose_solve(const csc_matrix_t<i_t, f_t>& U, std::vector<f_t>& x);

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
          std::vector<i_t>& xi);

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
                       typename std::vector<i_t>::iterator pstack);

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
                          f_t* x);

}  // namespace cuopt::linear_programming::dual_simplex
