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

#include <dual_simplex/triangle_solve.hpp>

#include <optional>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t lower_triangular_solve(const csc_matrix_t<i_t, f_t>& L, std::vector<f_t>& x)
{
  i_t n = L.n;
  assert(x.size() == n);
  for (i_t j = 0; j < n; ++j) {
    i_t col_start = L.col_start[j];
    i_t col_end   = L.col_start[j + 1];
    if (x[j] != 0.0) {
      x[j] /= L.x[col_start];
      for (i_t p = col_start + 1; p < col_end; ++p) {
        x[L.i[p]] -= L.x[p] * x[j];
      }
    }
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t lower_triangular_transpose_solve(const csc_matrix_t<i_t, f_t>& L, std::vector<f_t>& x)
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
  return 0;
}

template <typename i_t, typename f_t>
i_t upper_triangular_solve(const csc_matrix_t<i_t, f_t>& U, std::vector<f_t>& x)
{
  const i_t n = U.n;
  assert(x.size() == n);
  for (i_t j = n - 1; j >= 0; --j) {
    const i_t col_start = U.col_start[j];
    const i_t col_end   = U.col_start[j + 1] - 1;
    if (x[j] != 0.0) {
      x[j] /= U.x[col_end];
      for (i_t p = col_start; p < col_end; ++p) {
        x[U.i[p]] -= U.x[p] * x[j];
      }
    }
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t upper_triangular_transpose_solve(const csc_matrix_t<i_t, f_t>& U, std::vector<f_t>& x)
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
  return 0;
}

// \brief Reach computes the reach of b in the graph of G
// \param[in] b - Sparse vector containing the rhs
// \param[in] pinv - inverse permuation vector
// \param[in, out] G - Sparse CSC matrix G. The column pointers of G are
// modified (but restored) during this call \param[out] xi  - stack of size 2*n.
// xi[top] .. xi[n-1] contains the reachable indicies \returns top - the size of
// the stack
template <typename i_t, typename f_t>
i_t reach(const sparse_vector_t<i_t, f_t>& b,
          const std::optional<std::vector<i_t>>& pinv,
          csc_matrix_t<i_t, f_t>& G,
          std::vector<i_t>& xi)
{
  const i_t m   = G.m;
  i_t top       = m;
  const i_t bnz = b.i.size();
  for (i_t p = 0; p < bnz; ++p) {
    if (!MARKED(G.col_start, b.i[p])) {  // start a DFS at unmarked node i
      top = depth_first_search(b.i[p], pinv, G, top, xi, xi.begin() + m);
    }
  }
  for (i_t p = top; p < m; ++p) {  // restore G
    MARK(G.col_start, xi[p]);
  }
  return top;
}

// \brief Performs a depth-first search starting from node j in the graph
// defined by G \param[in] j - root node \param[in] pinv - inverse permutation
// \param[in, out] G - graph defined by sparse CSC matrix
// \param[in, out] top - top of the stack in xi
// \param[in, out] xi  - stack containing the nodes found in topological order
// \parma[in, out] pstack - private stack (points into xi)
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
                       typename std::vector<i_t>::iterator pstack)
{
  i_t head = 0;
  xi[0]    = j;  // Initialize the recursion stack
  i_t done = 0;
  while (head >= 0) {
    j        = xi[head];  // Get j from the top of the recursion stack
    i_t jnew = pinv ? ((*pinv)[j]) : j;
    if (!MARKED(G.col_start, j)) {
      // If node j is not marked this is the first time it has been visited
      MARK(G.col_start, j)  // Mark node j as visited
      // Point to the first outgoing edge of node j
      pstack[head] = (jnew < 0) ? 0 : UNFLIP(G.col_start[jnew]);
    }
    done   = 1;  // Node j is done if no unvisited neighbors
    i_t p2 = (jnew < 0) ? 0 : UNFLIP(G.col_start[jnew + 1]);
    for (i_t p = pstack[head]; p < p2; ++p) {  // Examine all neighbors of j
      i_t i = G.i[p];                          // Consider neighbor i
      if (MARKED(G.col_start, i)) {
        continue;  // skip visited node i
      }
      pstack[head] = p;  // pause depth-first search of node j
      xi[++head]   = i;  // start dfs at node i
      done         = 0;  // node j is not done
      break;             // break to start dfs at node i
    }
    if (done) {
      pstack[head] = 0;  // restore pstack so it can be used again in other routines
      xi[head]     = 0;  // restore xi so it can be used again in other routines
      head--;            // remove j from the recursion stack
      xi[--top] = j;     // and place it the output stack
    }
  }
  return top;
}

template <typename i_t, typename f_t, bool lo>
i_t sparse_triangle_solve(const sparse_vector_t<i_t, f_t>& b,
                          const std::optional<std::vector<i_t>>& pinv,
                          std::vector<i_t>& xi,
                          csc_matrix_t<i_t, f_t>& G,
                          f_t* x)
{
  i_t m = G.m;
  assert(b.n == m);
  i_t top = reach(b, pinv, G, xi);
  for (i_t p = top; p < m; ++p) {
    x[xi[p]] = 0;  // Clear x vector
  }
  const i_t bnz = b.i.size();
  for (i_t p = 0; p < bnz; ++p) {
    x[b.i[p]] = b.x[p];  // Scatter b
  }
  for (i_t px = top; px < m; ++px) {
    i_t j = xi[px];                   // x(j) is nonzero
    i_t J = pinv ? ((*pinv)[j]) : j;  // j maps to column J of G
    if (J < 0) continue;              // column j is empty
    f_t Gjj;
    i_t p;
    i_t end;
    if constexpr (lo) {
      Gjj = G.x[G.col_start[J]];  // lo: L(j, j) is the first entry
      p   = G.col_start[J] + 1;
      end = G.col_start[J + 1];
    } else {
      Gjj = G.x[G.col_start[J + 1] - 1];  // up: U(j,j) is the last entry
      p   = G.col_start[J];
      end = G.col_start[J + 1] - 1;
    }
    x[j] /= Gjj;
    for (; p < end; ++p) {
      x[G.i[p]] -= G.x[p] * x[j];  // x(i) -= G(i,j) * x(j)
    }
  }
  return top;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template int lower_triangular_solve<int, double>(const csc_matrix_t<int, double>& L,
                                                 std::vector<double>& x);

template int lower_triangular_transpose_solve<int, double>(const csc_matrix_t<int, double>& L,
                                                           std::vector<double>& x);

template int upper_triangular_solve<int, double>(const csc_matrix_t<int, double>& U,
                                                 std::vector<double>& x);

template int upper_triangular_transpose_solve<int, double>(const csc_matrix_t<int, double>& U,
                                                           std::vector<double>& x);

template int reach<int, double>(const sparse_vector_t<int, double>& b,
                                const std::optional<std::vector<int>>& pinv,
                                csc_matrix_t<int, double>& G,
                                std::vector<int>& xi);

template int depth_first_search<int, double>(int j,
                                             const std::optional<std::vector<int>>& pinv,
                                             csc_matrix_t<int, double>& G,
                                             int top,
                                             std::vector<int>& xi,
                                             std::vector<int>::iterator pstack);

template int sparse_triangle_solve<int, double, true>(const sparse_vector_t<int, double>& b,
                                                      const std::optional<std::vector<int>>& pinv,
                                                      std::vector<int>& xi,
                                                      csc_matrix_t<int, double>& G,
                                                      double* x);

template int sparse_triangle_solve<int, double, false>(const sparse_vector_t<int, double>& b,
                                                       const std::optional<std::vector<int>>& pinv,
                                                       std::vector<int>& xi,
                                                       csc_matrix_t<int, double>& G,
                                                       double* x);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
