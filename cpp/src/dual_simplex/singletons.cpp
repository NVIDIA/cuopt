/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/singletons.hpp>
#include <dual_simplex/triangle_solve.hpp>
#include <dual_simplex/types.hpp>

#include <cstdio>

namespace cuopt::linear_programming::dual_simplex {

// Destroys the queue but prints it
template <typename i_t>
void print_queue(std::queue<i_t>& q)
{
  printf("queue ");
  while (!q.empty()) {
    printf("%d ", q.front());
    q.pop();
  }
  printf("\n");
}

template <typename i_t, typename f_t>
i_t order_singletons(std::queue<i_t>& singleton_queue,
                     i_t& singletons_found,
                     row_col_graph_t<i_t>& G,
                     f_t& work_estimate)
{
  constexpr i_t kEmpty = -1;
  while (!singleton_queue.empty()) {
#ifdef PARANOID
    std::queue<Int> queue_copy = singleton_queue;
    PrintQueue(queue_copy);
#endif

    i_t xpivot = singleton_queue.front();
    singleton_queue.pop();
    assert(G.Xdeg[xpivot] >= 0);
    if (G.Xdeg[xpivot] != 1) {
      assert(G.Xdeg[xpivot] == 0);
      continue;
    }

#ifndef NDEBUG
    i_t deg = 0;
    i_t p2  = G.Xp[xpivot + 1];
    for (i_t p = G.Xp[xpivot]; p < p2; ++p) {
      i_t y = G.Xi[p];
      if (G.Ydeg[y] >= 0) { deg++; }
    }
    assert(deg == 1);
#endif

    // find the ypivot to match
    // there must only be one
    i_t ypivot = kEmpty;
    i_t xend   = G.Xp[xpivot + 1];
    for (i_t p = G.Xp[xpivot]; p < xend; ++p) {
      i_t y = G.Xi[p];
      if (G.Ydeg[y] >= 0) {
        ypivot = y;
        break;
      }
    }
    work_estimate += 2 * (xend - G.Xp[xpivot]);
    assert(ypivot != kEmpty);
    assert(G.Ydeg[ypivot] >= 0);

    // Decrement the degrees after removing the pivot
    i_t yend = G.Yp[ypivot + 1];
    for (i_t p = G.Yp[ypivot]; p < yend; ++p) {
      i_t x = G.Yi[p];
      if (G.Xdeg[x] < 0) continue;  // This x already eliminated
      assert(G.Xdeg[x] > 0);
      if (x == xpivot) continue;  // This x about to be eliminated
      i_t degree = --G.Xdeg[x];
      assert(G.Xdeg[x] >= 0);
      if (degree == 1) {
        // This is a new singleton put it at the end of the queue
        singleton_queue.push(x);
      }
    }
    work_estimate += 2 * (yend - G.Yp[ypivot]);
    // Mark the pivot by flipping the degrees
    G.Xdeg[xpivot] = FLIP(1);
    G.Ydeg[ypivot] = FLIP(G.Ydeg[ypivot]);

    // Put the pivots in the permuation vectors
    G.Xperm[singletons_found] = xpivot;
    G.Yperm[singletons_found] = ypivot;
    singletons_found++;
  }
  return singletons_found;
}

// \param [in,out]  workspace - size m
template <typename i_t, typename f_t>
void create_row_representation(const csc_matrix_t<i_t, f_t>& A,
                               std::vector<i_t>& row_start,
                               std::vector<i_t>& col_index,
                               std::vector<i_t>& workspace,
                               f_t& work_estimate)
{
  // row counts
  i_t n  = A.n;
  i_t m  = A.m;
  i_t nz = A.col_start[n];

  assert(workspace.size() == m);
  // Clear workspace
  for (i_t i = 0; i < m; ++i) {
    workspace[i] = 0;
  }
  work_estimate += m;

  // Compute row degrees
  for (i_t p = 0; p < nz; ++p) {
    workspace[A.i[p]]++;
  }
  work_estimate += 3 * nz;

  // Compute rowstart and overwrite workspace
  cumulative_sum(workspace, row_start);
  work_estimate += 4 * workspace.size();

  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      i_t q        = workspace[A.i[p]]++;
      col_index[q] = j;
    }
  }
  work_estimate += 2 * n + 4 * nz;
}

// Complete the permuation
template <typename i_t>
i_t complete_permutation(i_t singletons, std::vector<i_t>& Xdeg, std::vector<i_t>& Xperm)
{
  i_t n = Xdeg.size();
  assert(Xperm.size() == n);
  i_t num_empty = 0;
  i_t start     = singletons;
  for (i_t k = 0; k < n; ++k) {
    i_t deg = Xdeg[k];
    if (deg == 0) {
      // this row/column is empty
      num_empty++;
      Xperm[n - num_empty] = k;
    } else if (deg > 0) {
      // this row/column is nonempty
      assert(start < n - num_empty);
      Xperm[start++] = k;
    } else {
      // this is a singleton row/column. It is already in the permutation
      Xdeg[k] = FLIP(deg);
    }
  }
  assert(start == n - num_empty);
  return num_empty;
}

template <typename i_t, typename f_t>
i_t find_singletons(const csc_matrix_t<i_t, f_t>& A,
                    i_t& row_singletons,
                    std::vector<i_t>& row_perm,
                    i_t& col_singletons,
                    std::vector<i_t>& col_perm,
                    f_t& work_estimate)
{
  i_t n  = A.n;
  i_t m  = A.m;
  i_t nz = A.col_start[n];

  std::vector<int> workspace(m);
  std::vector<i_t> Rdeg(m);
  std::vector<i_t> Cdeg(n);
  std::vector<i_t> Rp(m + 1);
  std::vector<i_t> Rj(nz);
  work_estimate += 3 * m + n + nz;

  i_t max_queue_len = std::max(m, n);
  std::queue<i_t> singleton_queue;

  // Compute Cdeg and Rdeg
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    Cdeg[j]             = col_end - col_start;
    for (i_t p = col_start; p < col_end; ++p) {
      Rdeg[A.i[p]]++;
    }
  }
  work_estimate += 2 * n + 2 * nz;

  // Add all columns of degree 1 to the queue
  for (i_t j = n - 1; j >= 0; --j) {
    if (Cdeg[j] == 1) { singleton_queue.push(j); }
  }
  work_estimate += n + singleton_queue.size();

  bool row_form        = false;
  i_t singletons_found = 0;
  col_singletons       = 0;
  if (!singleton_queue.empty()) {
    // Don't create the row representation unless we found a singleton
    create_row_representation(A, Rp, Rj, workspace, work_estimate);
    row_form = true;

    // Find column singletons
    row_col_graph_t<i_t> graph{Cdeg.begin(),
                               col_perm.begin(),
                               A.col_start.cbegin(),
                               A.i.cbegin(),
                               Rdeg.begin(),
                               row_perm.begin(),
                               Rp.cbegin(),
                               Rj.cbegin()};

#ifdef SINGLETON_DEBUG
    printf("Searching for column singletons. Initial size %ld\n", singleton_queue.size());
#endif

    col_singletons = order_singletons(singleton_queue, singletons_found, graph, work_estimate);

#ifdef SINGLETON_DEBUG
    printf("Found %d column singletons\n", col_singletons);
#endif
  }

  // Add all rows of degree 1 to the quee
  for (i_t i = m - 1; i >= 0; --i) {
    if (Rdeg[i] == 1) { singleton_queue.push(i); }
  }
  work_estimate += m + singleton_queue.size();

  row_singletons = 0;
  if (!singleton_queue.empty()) {
    if (!row_form) {
      // If we haven't created the row representation yet, we need to
      create_row_representation(A, Rp, Rj, workspace, work_estimate);  // use Rdeg as workspace
    }

    // Find row singletons
    row_col_graph_t<i_t> graph{Rdeg.begin(),
                               row_perm.begin(),
                               Rp.cbegin(),
                               Rj.cbegin(),
                               Cdeg.begin(),
                               col_perm.begin(),
                               A.col_start.cbegin(),
                               A.i.cbegin()};
#ifdef SINGLETON_DEBUG
    printf("Searching for row singletons %ld\n", singleton_queue.size());
#endif
    i_t last       = singletons_found;
    row_singletons = order_singletons(singleton_queue, singletons_found, graph, work_estimate);
    row_singletons = row_singletons - last;
#ifdef SINGLETON_DEBUG
    printf("Found %d row singletons. %d\n", row_singletons, singletons_found);
#endif
  } else {
#ifdef SINGLETON_DEBUG
    printf("No row singletons\n");
#endif
  }

#ifdef SINGLETON_DEBUG
  printf("Col singletons %d\n", col_singletons);
#endif
  i_t num_empty_cols = complete_permutation(singletons_found, Cdeg, col_perm);
  work_estimate += 2 * Cdeg.size();
#ifdef SINGLETON_DEBUG
  printf("Completed col perm. %d empty cols. Starting row perm\n", num_empty_cols);
#endif
  i_t num_empty_rows = complete_permutation(singletons_found, Rdeg, row_perm);
  work_estimate += 2 * Rdeg.size();
#ifdef SINGLETON_DEBUG
  printf("Empty rows %d Empty columns %d\n", num_empty_rows, num_empty_cols);
#endif
  return singletons_found;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template struct row_col_graph_t<int>;

template int order_singletons<int, double>(std::queue<int>& singleton_queue,
                                           int& singletons_found,
                                           row_col_graph_t<int>& G,
                                           double& work_estimate);

// \param [in,out]  workspace - size m
template void create_row_representation<int, double>(const csc_matrix_t<int, double>& A,
                                                     std::vector<int>& row_start,
                                                     std::vector<int>& col_index,
                                                     std::vector<int>& workspace,
                                                     double& work_estimate);
// Complete the permuation
template int complete_permutation<int>(int singletons,
                                       std::vector<int>& Xdeg,
                                       std::vector<int>& Xperm);

template int find_singletons<int, double>(const csc_matrix_t<int, double>& A,
                                          int& row_singletons,
                                          std::vector<int>& row_perm,
                                          int& col_singleton,
                                          std::vector<int>& col_perm,
                                          double& work_estimate);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
