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

#include <dual_simplex/right_looking_lu.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>

namespace cuopt::linear_programming::dual_simplex {

namespace {

// An element_t structure holds the information associated with a coefficient in the active
// submatrix during the LU factorization
template <typename i_t, typename f_t>
struct element_t {
  i_t i;               // row index
  i_t j;               // column index
  f_t x;               // coefficient value
  i_t next_in_column;  // index of the next element in the column: kNone if there is no next element
  i_t next_in_row;     // index of the next element in the row: kNone if there is no next element
};
constexpr int kNone = -1;

template <typename i_t, typename f_t>
i_t initialize_degree_data(const csc_matrix_t<i_t, f_t>& A,
                           const std::vector<i_t>& column_list,
                           std::vector<i_t>& Cdegree,
                           std::vector<i_t>& Rdegree,
                           std::vector<std::vector<i_t>>& col_count,
                           std::vector<std::vector<i_t>>& row_count)
{
  const i_t n = column_list.size();
  const i_t m = A.m;
  std::fill(Rdegree.begin(), Rdegree.end(), 0);
  i_t Bnz = 0;
  for (i_t k = 0; k < n; ++k) {
    const i_t j         = column_list[k];
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    Cdegree[k]          = col_end - col_start;
    for (i_t p = col_start; p < col_end; ++p) {
      Rdegree[A.i[p]]++;
      Bnz++;
    }
  }

  for (i_t k = 0; k < n; ++k) {
    assert(Cdegree[k] <= n && Cdegree[k] >= 0);
    col_count[Cdegree[k]].push_back(k);
  }

  for (i_t k = 0; k < m; ++k) {
    assert(Rdegree[k] <= m && Rdegree[k] >= 0);
    row_count[Rdegree[k]].push_back(k);
    if (Rdegree[k] == 0) {
      constexpr bool verbose = false;
      if (verbose) { printf("Zero degree row %d\n", k); }
    }
  }
  return Bnz;
}

template <typename i_t, typename f_t>
i_t load_elements(const csc_matrix_t<i_t, f_t>& A,
                  const std::vector<i_t>& column_list,
                  i_t Bnz,
                  std::vector<element_t<i_t, f_t>>& elements,
                  std::vector<i_t>& first_in_row,
                  std::vector<i_t>& first_in_col)
{
  const i_t m = A.m;
  const i_t n = column_list.size();
  std::vector<i_t> last_element_in_row(m, kNone);

  i_t nz = 0;
  for (i_t k = 0; k < n; ++k) {
    const i_t j         = column_list[k];
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      const i_t i                 = A.i[p];
      elements[nz].i              = i;
      elements[nz].j              = k;
      elements[nz].x              = A.x[p];
      elements[nz].next_in_column = kNone;
      if (p > col_start) { elements[nz - 1].next_in_column = nz; }
      elements[nz].next_in_row = kNone;  // set the current next in row to None (since we don't know
                                         // if there will be more entries in this row)
      if (last_element_in_row[i] != kNone) {
        // If we have seen an entry in this row before, set the last entry we've seen in this row to
        // point to the current entry
        elements[last_element_in_row[i]].next_in_row = nz;
      }
      // The current entry becomes the last element seen in the row
      last_element_in_row[i] = nz;
      if (p == col_start) { first_in_col[k] = nz; }
      if (first_in_row[i] == kNone) { first_in_row[i] = nz; }
      nz++;
    }
  }
  assert(nz == Bnz);

  for (i_t j = 0; j < n; j++) {
    for (i_t p = first_in_col[j]; p != kNone; p = elements[p].next_in_column) {
      element_t<i_t, f_t>* entry = &elements[p];
      assert(entry->j == j);
      assert(entry->i >= 0);
      assert(entry->i < m);
    }
  }

  for (i_t i = 0; i < m; i++) {
    for (i_t p = first_in_row[i]; p != kNone; p = elements[p].next_in_row) {
      element_t<i_t, f_t>* entry = &elements[p];
      assert(entry->i == i);
      assert(entry->j < n);
      assert(entry->j >= 0);
    }
  }

  return 0;
}

template <typename i_t, typename f_t>
f_t maximum_in_column(i_t j,
                      const std::vector<i_t>& first_in_col,
                      std::vector<element_t<i_t, f_t>>& elements)
{
  f_t max_in_col = 0.0;
  for (i_t p = first_in_col[j]; p != kNone; p = elements[p].next_in_column) {
    element_t<i_t, f_t>* entry = &elements[p];
    assert(entry->j == j);
    max_in_col = std::max(max_in_col, std::abs(entry->x));
  }
  return max_in_col;
}

template <typename i_t, typename f_t>
void initialize_max_in_column(const std::vector<i_t>& first_in_col,
                              std::vector<element_t<i_t, f_t>>& elements,
                              std::vector<f_t>& max_in_column)
{
  const i_t n = first_in_col.size();
  for (i_t j = 0; j < n; ++j) {
    max_in_column[j] = maximum_in_column(j, first_in_col, elements);
  }
}

template <typename i_t, typename f_t>
f_t maximum_in_row(i_t i,
                   const std::vector<i_t>& first_in_row,
                   std::vector<element_t<i_t, f_t>>& elements)
{
  f_t max_in_row = 0.0;
  for (i_t p = first_in_row[i]; p != kNone; p = elements[p].next_in_row) {
    element_t<i_t, f_t>* entry = &elements[p];
    assert(entry->i == i);
    max_in_row = std::max(max_in_row, std::abs(entry->x));
  }
  return max_in_row;
}

template <typename i_t, typename f_t>
void initialize_max_in_row(const std::vector<i_t>& first_in_row,
                           std::vector<element_t<i_t, f_t>>& elements,
                           std::vector<f_t>& max_in_row)
{
  const i_t m = first_in_row.size();
  for (i_t i = 0; i < m; ++i) {
    max_in_row[i] = maximum_in_row(i, first_in_row, elements);
  }
}

#undef THRESHOLD_ROOK_PIVOTING  // Disable threshold rook pivoting for now.
                                // 3% slower when enabled. But keep it around
                                // for challenging numerical problems.
template <typename i_t, typename f_t>
i_t markowitz_search(const std::vector<i_t>& Cdegree,
                     const std::vector<i_t>& Rdegree,
                     const std::vector<std::vector<i_t>>& col_count,
                     const std::vector<std::vector<i_t>>& row_count,
                     const std::vector<i_t>& first_in_row,
                     const std::vector<i_t>& first_in_col,
                     const std::vector<f_t>& max_in_column,
                     const std::vector<f_t>& max_in_row,
                     std::vector<element_t<i_t, f_t>>& elements,
                     f_t pivot_tol,
                     f_t threshold_tol,
                     i_t& pivot_i,
                     i_t& pivot_j,
                     i_t& pivot_p)
{
  i_t nz      = 1;
  const i_t m = Rdegree.size();
  const i_t n = Cdegree.size();
  f_t markowitz =
    static_cast<f_t>(m) * static_cast<f_t>(n);  // upper bound on largest markowtiz criteria
  i_t nsearch            = 0;
  constexpr bool verbose = false;
  i_t nz_max             = std::min(m, n);
  while (nz <= nz_max) {
    i_t markowitz_lower_bound = (nz - 1) * (nz - 1);
    // Search columns of length nz
    for (const i_t j : col_count[nz]) {
      assert(Cdegree[j] == nz);
      const f_t max_in_col = max_in_column[j];

      for (i_t p = first_in_col[j]; p != kNone; p = elements[p].next_in_column) {
        element_t<i_t, f_t>* entry = &elements[p];
        const i_t i                = entry->i;
        assert(entry->j == j);
#ifdef CHECK_RDEGREE
        if (Rdegree[i] < 0) {
          if (verbose) {
            printf("Rdegree[%d] %d. Searching in column %d. Entry i %d j %d val %e\n",
                   i,
                   Rdegree[i],
                   j,
                   entry->i,
                   entry->j,
                   entry->x);
          }
        }
#endif
        assert(Rdegree[i] >= 0);
        const i_t Mij = (Rdegree[i] - 1) * (nz - 1);
        if (Mij < markowitz && std::abs(entry->x) >= threshold_tol * max_in_col &&
#ifdef THRESHOLD_ROOK_PIVOTING
            std::abs(entry->x) >= threshold_tol * max_in_row[i] &&
#endif
            std::abs(entry->x) >= pivot_tol) {
          markowitz = Mij;
          pivot_i   = i;
          pivot_j   = j;
          pivot_p   = p;
          if (markowitz <= markowitz_lower_bound) { break; }
        }
      }
      nsearch++;
      if (markowitz <= markowitz_lower_bound) { break; }
    }

    if (markowitz <= markowitz_lower_bound) { break; }

    markowitz_lower_bound = (nz - 1) * nz;

    // Search rows of length nz
    assert(row_count[nz].size() >= 0);
    for (const i_t i : row_count[nz]) {
      assert(Rdegree[i] == nz);
#ifdef THRESHOLD_ROOK_PIVOTING
      const f_t max_in_row_i = max_in_row[i];
#endif
      for (i_t p = first_in_row[i]; p != kNone; p = elements[p].next_in_row) {
        element_t<i_t, f_t>* entry = &elements[p];
        const i_t j                = entry->j;
        assert(entry->i == i);
        const f_t max_in_col = max_in_column[j];
        assert(Cdegree[j] >= 0);
        const i_t Mij = (nz - 1) * (Cdegree[j] - 1);
        if (Mij < markowitz && std::abs(entry->x) >= threshold_tol * max_in_col &&
#ifdef THRESHOLD_ROOK_PIVOTING
            std::abs(entry->x) >= threshold_tol * max_in_row_i &&
#endif
            std::abs(entry->x) >= pivot_tol) {
          markowitz = Mij;
          pivot_i   = i;
          pivot_j   = j;
          pivot_p   = p;
          if (markowitz <= markowitz_lower_bound) { break; }
        }
      }
      nsearch++;
      if (markowitz <= markowitz_lower_bound) { break; }
    }

    if (pivot_i != -1 && nz >= 2) { break; }
    nz++;
  }
  if (nsearch > 10) {
    if constexpr (verbose) { printf("nsearch %d\n", nsearch); }
  }
  return nsearch;
}

template <typename i_t, typename f_t>
void update_Cdegree_and_col_count(i_t pivot_i,
                                  i_t pivot_j,
                                  const std::vector<i_t>& first_in_row,
                                  std::vector<i_t>& Cdegree,
                                  std::vector<std::vector<i_t>>& col_count,
                                  std::vector<element_t<i_t, f_t>>& elements)
{
  // Update Cdegree and col_count
  for (i_t p = first_in_row[pivot_i]; p != kNone; p = elements[p].next_in_row) {
    element_t<i_t, f_t>* entry = &elements[p];
    const i_t j                = entry->j;
    assert(entry->i == pivot_i);
    i_t cdeg = Cdegree[j];
    assert(cdeg >= 0);
    for (typename std::vector<i_t>::iterator it = col_count[cdeg].begin();
         it != col_count[cdeg].end();
         it++) {
      if (*it == j) {
        // Remove col j from col_count[cdeg]
        std::swap(*it, col_count[cdeg].back());
        col_count[cdeg].pop_back();
        break;
      }
    }
    cdeg = --Cdegree[j];
    assert(cdeg >= 0);
    if (j != pivot_j && cdeg >= 0) { col_count[cdeg].push_back(j); }
  }
  Cdegree[pivot_j] = -1;
}

template <typename i_t, typename f_t>
void update_Rdegree_and_row_count(i_t pivot_i,
                                  i_t pivot_j,
                                  const std::vector<i_t>& first_in_col,
                                  std::vector<i_t>& Rdegree,
                                  std::vector<std::vector<i_t>>& row_count,
                                  std::vector<element_t<i_t, f_t>>& elements)
{
  // Update Rdegree and row_count
  for (i_t p = first_in_col[pivot_j]; p != kNone; p = elements[p].next_in_column) {
    element_t<i_t, f_t>* entry = &elements[p];
    const i_t i                = entry->i;
    i_t rdeg                   = Rdegree[i];
    assert(rdeg >= 0);
    for (typename std::vector<i_t>::iterator it = row_count[rdeg].begin();
         it != row_count[rdeg].end();
         it++) {
      if (*it == i) {
        // Remove row i from row_count[rdeg]
        std::swap(*it, row_count[rdeg].back());
        row_count[rdeg].pop_back();
        break;
      }
    }
    rdeg = --Rdegree[i];
    assert(rdeg >= 0);
    if (i != pivot_i && rdeg >= 0) { row_count[rdeg].push_back(i); }
  }
  Rdegree[pivot_i] = -1;
}

template <typename i_t, typename f_t>
void schur_complement(i_t pivot_i,
                      i_t pivot_j,
                      f_t drop_tol,
                      f_t pivot_val,
                      i_t pivot_p,
                      element_t<i_t, f_t>*& pivot_entry,
                      std::vector<i_t>& first_in_col,
                      std::vector<i_t>& first_in_row,
                      std::vector<i_t>& row_last_workspace,
                      std::vector<i_t>& column_j_workspace,
                      std::vector<f_t>& max_in_column,
                      std::vector<f_t>& max_in_row,
                      std::vector<i_t>& Rdegree,
                      std::vector<i_t>& Cdegree,
                      std::vector<std::vector<i_t>>& row_count,
                      std::vector<std::vector<i_t>>& col_count,
                      std::vector<element_t<i_t, f_t>>& elements)
{
  for (i_t p1 = first_in_col[pivot_j]; p1 != kNone; p1 = elements[p1].next_in_column) {
    element_t<i_t, f_t>* e = &elements[p1];
    const i_t i            = e->i;
    i_t row_last           = kNone;
    for (i_t p3 = first_in_row[i]; p3 != kNone; p3 = elements[p3].next_in_row) {
      row_last = p3;
    }
    row_last_workspace[i] = row_last;
  }

  for (i_t p0 = first_in_row[pivot_i]; p0 != kNone; p0 = elements[p0].next_in_row) {
    element_t<i_t, f_t>* entry = &elements[p0];
    const i_t j                = entry->j;
    assert(entry->i == pivot_i);
    if (j == pivot_j) { continue; }
    const f_t uj = entry->x;

    i_t col_last = kNone;
    for (i_t p1 = first_in_col[j]; p1 != kNone; p1 = elements[p1].next_in_column) {
      element_t<i_t, f_t>* e = &elements[p1];
      const i_t i            = e->i;
      assert(e->j == j);
      column_j_workspace[i] = p1;
      col_last              = p1;
    }

    for (i_t p1 = first_in_col[pivot_j]; p1 != kNone; p1 = elements[p1].next_in_column) {
      element_t<i_t, f_t>* e = &elements[p1];
      const i_t i            = e->i;
      assert(e->j == pivot_j);
      if (i == pivot_i) { continue; }
      const f_t li  = e->x / pivot_val;
      const f_t val = li * uj;
      if (std::abs(val) < drop_tol) { continue; }
      if (column_j_workspace[i] != kNone) {
        element_t<i_t, f_t>* e2 = &elements[column_j_workspace[i]];
        e2->x -= val;
        const f_t abs_e2x = std::abs(e2->x);
        if (abs_e2x > max_in_column[j]) { max_in_column[j] = abs_e2x; }
#ifdef THRESHOLD_ROOK_PIVOTING
        if (abs_e2x > max_in_row[i]) { max_in_row[i] = abs_e2x; }
#endif
      } else {
        element_t<i_t, f_t> fill;
        fill.i              = i;
        fill.j              = j;
        fill.x              = -val;
        const f_t abs_fillx = std::abs(fill.x);
        if (abs_fillx > max_in_column[j]) { max_in_column[j] = abs_fillx; }
#ifdef THRESHOLD_ROOK_PIVOTING
        if (abs_fillx > max_in_row[i]) { max_in_row[i] = abs_fillx; }
#endif
        fill.next_in_column = kNone;
        fill.next_in_row    = kNone;
        elements.push_back(fill);
        pivot_entry =
          &elements[pivot_p];  // push_back could cause a realloc so need to get a new pointer
        i_t fill_p = elements.size() - 1;
        assert(elements[fill_p].x == fill.x);
        if (col_last != kNone) {
          elements[col_last].next_in_column = fill_p;
        } else {
          first_in_col[j] = fill_p;
        }
        col_last     = fill_p;
        i_t row_last = row_last_workspace[i];
        if (row_last != kNone) {
          elements[row_last].next_in_row = fill_p;
        } else {
          first_in_row[i] = fill_p;
        }
        row_last_workspace[i] = fill_p;
        i_t rdeg              = Rdegree[i];  // Rdgree must increase
        for (typename std::vector<i_t>::iterator it = row_count[rdeg].begin();
             it != row_count[rdeg].end();
             it++) {
          if (*it == i) {
            // Remove row i from row_count[rdeg]
            std::swap(*it, row_count[rdeg].back());
            row_count[rdeg].pop_back();
            break;
          }
        }
        rdeg = ++Rdegree[i];           // Increase rdeg
        row_count[rdeg].push_back(i);  // Add row i to row_count[rdeg]

        i_t cdeg = Cdegree[j];  // Cdegree must increase
        for (typename std::vector<i_t>::iterator it = col_count[cdeg].begin();
             it != col_count[cdeg].end();
             it++) {
          if (*it == j) {
            // Remove col j from col_count[cdeg]
            std::swap(*it, col_count[cdeg].back());
            col_count[cdeg].pop_back();
            break;
          }
        }
        cdeg = ++Cdegree[j];           // Increase Cdegree
        col_count[cdeg].push_back(j);  // Add column j to col_count[cdeg]
      }
    }

    for (i_t p1 = first_in_col[j]; p1 != kNone; p1 = elements[p1].next_in_column) {
      element_t<i_t, f_t>* e = &elements[p1];
      const i_t i            = e->i;
      assert(e->j == j);
      column_j_workspace[i] = kNone;
    }
  }
}

template <typename i_t, typename f_t>
void remove_pivot_row(i_t pivot_i,
                      i_t pivot_j,
                      std::vector<i_t>& first_in_col,
                      std::vector<i_t>& first_in_row,
                      std::vector<f_t>& max_in_column,
                      std::vector<element_t<i_t, f_t>>& elements)
{
  // Remove the pivot row

  for (i_t p0 = first_in_row[pivot_i]; p0 != kNone; p0 = elements[p0].next_in_row) {
    element_t<i_t, f_t>* e = &elements[p0];
    const i_t j            = e->j;
    if (j == pivot_j) { continue; }
    i_t last         = kNone;
    f_t max_in_col_j = 0;
    for (i_t p = first_in_col[j]; p != kNone; p = elements[p].next_in_column) {
      element_t<i_t, f_t>* entry = &elements[p];
      if (entry->i == pivot_i) {
        if (last != kNone) {
          elements[last].next_in_column = entry->next_in_column;
        } else {
          first_in_col[j] = entry->next_in_column;
        }
        entry->i = -1;
        entry->j = -1;
        entry->x = std::numeric_limits<f_t>::quiet_NaN();
      } else {
        const f_t abs_entryx = std::abs(entry->x);
        if (abs_entryx > max_in_col_j) { max_in_col_j = abs_entryx; }
      }
      last = p;
    }
    max_in_column[j] = max_in_col_j;
  }

  first_in_row[pivot_i] = kNone;
}

template <typename i_t, typename f_t>
void remove_pivot_col(i_t pivot_i,
                      i_t pivot_j,
                      std::vector<i_t>& first_in_col,
                      std::vector<i_t>& first_in_row,
                      std::vector<f_t>& max_in_row,
                      std::vector<element_t<i_t, f_t>>& elements)
{
  // Remove the pivot col
  for (i_t p1 = first_in_col[pivot_j]; p1 != kNone; p1 = elements[p1].next_in_column) {
    element_t<i_t, f_t>* e = &elements[p1];
    const i_t i            = e->i;
    i_t last               = kNone;
#ifdef THRESHOLD_ROOK_PIVOTING
    f_t max_in_row_i = 0.0;
#endif
    for (i_t p = first_in_row[i]; p != kNone; p = elements[p].next_in_row) {
      element_t<i_t, f_t>* entry = &elements[p];
      if (entry->j == pivot_j) {
        if (last != kNone) {
          elements[last].next_in_row = entry->next_in_row;
        } else {
          first_in_row[i] = entry->next_in_row;
        }
        entry->i = -1;
        entry->j = -1;
        entry->x = std::numeric_limits<f_t>::quiet_NaN();
      }
#ifdef THRESHOLD_ROOK_PIVOTING
      else {
        const f_t abs_entryx = std::abs(entry->x);
        if (abs_entryx > max_in_row_i) { max_in_row_i = abs_entryx; }
      }
#endif
      last = p;
    }
#ifdef THRESHOLD_ROOK_PIVOTING
    max_in_row[i] = max_in_row_i;
#endif
  }
  first_in_col[pivot_j] = kNone;
}

}  // namespace

template <typename i_t, typename f_t>
i_t right_looking_lu(const csc_matrix_t<i_t, f_t>& A,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     f_t tol,
                     const std::vector<i_t>& column_list,
                     std::vector<i_t>& q,
                     csc_matrix_t<i_t, f_t>& L,
                     csc_matrix_t<i_t, f_t>& U,
                     std::vector<i_t>& pinv)
{
  const i_t n = column_list.size();
  const i_t m = A.m;

  assert(A.m == n);
  assert(L.n == n);
  assert(L.m == n);
  assert(U.n == n);
  assert(U.m == n);
  assert(q.size() == n);
  assert(pinv.size() == n);

  std::vector<i_t> Rdegree(n);  // Rdegree[i] is the degree of row i
  std::vector<i_t> Cdegree(n);  // Cdegree[j] is the degree of column j

  std::vector<std::vector<i_t>> col_count(
    n + 1);  // col_count[nz] is a list of columns with nz nonzeros in the active submatrix
  std::vector<std::vector<i_t>> row_count(
    n + 1);  // row_count[nz] is a list of rows with nz nonzeros in the active submatrix

  const i_t Bnz = initialize_degree_data(A, column_list, Cdegree, Rdegree, col_count, row_count);
  std::vector<element_t<i_t, f_t>> elements(Bnz);
  std::vector<i_t> first_in_row(n, kNone);
  std::vector<i_t> first_in_col(n, kNone);
  load_elements(A, column_list, Bnz, elements, first_in_row, first_in_col);

  std::vector<i_t> column_j_workspace(n, kNone);
  std::vector<i_t> row_last_workspace(n);
  std::vector<f_t> max_in_column(n);
  std::vector<f_t> max_in_row(m);
  initialize_max_in_column(first_in_col, elements, max_in_column);
#ifdef THRESHOLD_ROOK_PIVOTING
  initialize_max_in_row(first_in_row, elements, max_in_row);
#endif

  csr_matrix_t<i_t, f_t> Urow(n, n, 0);  // We will store U by rows in Urow during the factorization
                                         // and translate back to U at the end
  Urow.n = Urow.m = n;
  Urow.row_start.resize(n + 1, -1);
  i_t Unz = 0;

  i_t Lnz = 0;
  L.x.clear();
  L.i.clear();

  std::fill(q.begin(), q.end(), -1);
  std::fill(pinv.begin(), pinv.end(), -1);
  std::vector<i_t> qinv(n);
  std::fill(qinv.begin(), qinv.end(), -1);

  i_t pivots = 0;
  for (i_t k = 0; k < n; ++k) {
    if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) { return -1; }
    // Find pivot that satisfies
    // abs(pivot) >= abstol,
    // abs(pivot) >= threshold_tol * max abs[pivot column]
    i_t pivot_i             = -1;
    i_t pivot_j             = -1;
    i_t pivot_p             = kNone;
    constexpr f_t pivot_tol = 1e-11;
    const f_t drop_tol      = tol == 1.0 ? 0.0 : 1e-13;
    const f_t threshold_tol = tol;
    markowitz_search(Cdegree,
                     Rdegree,
                     col_count,
                     row_count,
                     first_in_row,
                     first_in_col,
                     max_in_column,
                     max_in_row,
                     elements,
                     pivot_tol,
                     threshold_tol,
                     pivot_i,
                     pivot_j,
                     pivot_p);
    if (pivot_i == -1 || pivot_j == -1) { break; }
    element_t<i_t, f_t>* pivot_entry = &elements[pivot_p];
    assert(pivot_i != -1 && pivot_j != -1);
    assert(pivot_entry->i == pivot_i && pivot_entry->j == pivot_j);

    // Pivot
    pinv[pivot_i]       = k;  // pivot_i is the kth pivot row
    q[k]                = pivot_j;
    qinv[pivot_j]       = k;
    const f_t pivot_val = pivot_entry->x;
    assert(std::abs(pivot_val) >= pivot_tol);
    pivots++;

    // U <- [U; u^T]
    Urow.row_start[k] = Unz;
    // U(k, pivot_j) = pivot_val
    Urow.j.push_back(pivot_j);
    Urow.x.push_back(pivot_val);
    Unz++;
    // U(k, :)
    for (i_t p = first_in_row[pivot_i]; p != kNone; p = elements[p].next_in_row) {
      element_t<i_t, f_t>* entry = &elements[p];
      const i_t j                = entry->j;
      assert(entry->i == pivot_i);
      if (j != pivot_j) {
        Urow.j.push_back(j);
        Urow.x.push_back(entry->x);
        Unz++;
      }
    }

    // L <- [L l]
    L.col_start[k] = Lnz;
    // L(pivot_i, k) = 1
    L.i.push_back(pivot_i);
    L.x.push_back(1.0);
    Lnz++;

    // L(:, k)
    for (i_t p = first_in_col[pivot_j]; p != kNone; p = elements[p].next_in_column) {
      element_t<i_t, f_t>* entry = &elements[p];
      const i_t i                = entry->i;
      assert(entry->j == pivot_j);
      if (i != pivot_i) {
        L.i.push_back(i);
        const f_t l_val = entry->x / pivot_val;
        L.x.push_back(l_val);
        Lnz++;
      }
    }

    // Update Cdegree and col_count
    update_Cdegree_and_col_count(pivot_i, pivot_j, first_in_row, Cdegree, col_count, elements);
    update_Rdegree_and_row_count(pivot_i, pivot_j, first_in_col, Rdegree, row_count, elements);

    // A22 <- A22 - l u^T
    schur_complement(pivot_i,
                     pivot_j,
                     drop_tol,
                     pivot_val,
                     pivot_p,
                     pivot_entry,
                     first_in_col,
                     first_in_row,
                     row_last_workspace,
                     column_j_workspace,
                     max_in_column,
                     max_in_row,
                     Rdegree,
                     Cdegree,
                     row_count,
                     col_count,
                     elements);

    // Remove the pivot row
    remove_pivot_row(pivot_i, pivot_j, first_in_col, first_in_row, max_in_column, elements);
    remove_pivot_col(pivot_i, pivot_j, first_in_col, first_in_row, max_in_row, elements);

    // Set pivot entry to sentinel value
    pivot_entry->i = -1;
    pivot_entry->j = -1;
    pivot_entry->x = std::numeric_limits<f_t>::quiet_NaN();

#ifdef CHECK_MAX_IN_COLUMN
    // Check that maximum in column is maintained
    for (i_t j = 0; j < n; ++j) {
      if (Cdegree[j] == -1) { continue; }
      const f_t max_in_col = max_in_column[j];
      bool found_max       = false;
      f_t largest_abs_x    = 0;
      for (i_t p = first_in_col[j]; p != kNone; p = elements[p].next_in_column) {
        const f_t abs_e2x = std::abs(elements[p].x);
        if (abs_e2x > largest_abs_x) { largest_abs_x = abs_e2x; }
        if (abs_e2x > max_in_col) {
          printf("Found max in column %d is %e but %e\n", j, max_in_col, abs_e2x);
        }
        assert(abs_e2x <= max_in_col);
        if (abs_e2x == max_in_col) { found_max = true; }
      }
      if (!found_max) {
        printf(
          "Did not find max %e in column %d. Largest abs x is %e\n", max_in_col, j, largest_abs_x);
      }
      assert(found_max);
    }
#endif

#ifdef CHECK_MAX_IN_ROW
    // Check that maximum in row is maintained
    for (i_t i = 0; i < m; ++i) {
      if (Rdegree[i] == -1) { continue; }
      const f_t max_in_row_i = max_in_row[i];
      bool found_max         = false;
      f_t largest_abs_x      = 0.0;
      for (i_t p = first_in_row[i]; p != kNone; p = elements[p].next_in_row) {
        const f_t abs_e2x = std::abs(elements[p].x);
        if (abs_e2x > largest_abs_x) { largest_abs_x = abs_e2x; }
        if (abs_e2x > max_in_row_i) {
          printf("Found max in row %d is %e but %e\n", i, max_in_row_i, abs_e2x);
        }
        assert(abs_e2x <= max_in_row_i);
        if (abs_e2x == max_in_row_i) { found_max = true; }
      }
      if (!found_max) {
        printf(
          "Did not find max %e in row %d. Largest abs x is %e\n", max_in_row_i, i, largest_abs_x);
      }
      assert(found_max);
    }
#endif

#if CHECK_BAD_ENTRIES
    for (Int j = 0; j < n; j++) {
      for (Int p = first_in_col[j]; p != kNone; p = elements[p].next_in_column) {
        element_t* entry = &elements[p];
        if (entry->i == -1) { printf("Found bad entry in row %d and column %d\n", entry->i, j); }
        assert(entry->i != -1);
        assert(entry->i != pivot_i);
        assert(entry->j != -1);
        assert(entry->j == j);
        assert(entry->j != pivot_j);
        assert(entry->x == entry->x);
      }
    }

    for (Int i = 0; i < n; i++) {
      for (Int p = first_in_row[i]; p != kNone; p = elements[p].next_in_row) {
        element_t* entry = &elements[p];
        if (entry->i == -1) {
          printf("Bad entry found in row %d. i %d j %d val %e\n", i, entry->i, entry->j, entry->x);
        }
        assert(entry->i != -1);
        assert(entry->i == i);
        assert(entry->i != pivot_i);
        assert(entry->j != -1);
        assert(entry->j != pivot_j);
        assert(entry->x == entry->x);
      }
    }
#endif

#ifdef WRITE_FACTORIZATION
    {
      FILE* file;
      if (k == 0) {
        file = fopen("factorization.m", "w");
      } else {
        file = fopen("factorization.m", "a");
      }
      if (file != NULL) {
        fprintf(file, "m = %d;\n", m);
        fprintf(file, "ijx = [\n");
        for (i_t j = 0; j < n; j++) {
          for (i_t p = first_in_col[j]; p != kNone; p = elements[p].next_in_column) {
            element_t<i_t, f_t>* e = &elements[p];
            fprintf(file, "%d %d %e;\n", e->i + 1, e->j + 1, e->x);
          }
        }
        fprintf(file, "];\n");
        fprintf(file, "if ~isempty(ijx)\n");
        fprintf(file, "B_%d = sparse(ijx(:, 1), ijx(:, 2), ijx(:,3), m, m);\n", k);
        fprintf(file, "end\n");
        fprintf(file, "pinv(%d) = %d;\n", pivot_i + 1, k + 1);
        fprintf(file, "q(%d) = %d;\n", k + 1, pivot_j + 1);
      }
      fclose(file);
    }
#endif
  }

  // Check for rank deficiency
  if (pivots < n) {
    // Complete the permutation pinv
    i_t start = pivots;
    for (i_t i = 0; i < m; ++i) {
      if (pinv[i] == -1) { pinv[i] = start++; }
    }

    // Finalize the permutation q. Do this by first completing the inverse permutation qinv.
    // Then invert qinv to get the final permutation q.
    start = pivots;
    for (i_t j = 0; j < n; ++j) {
      if (qinv[j] == -1) { qinv[j] = start++; }
    }
    inverse_permutation(qinv, q);

    return pivots;
  }

  // Finalize L and Urow
  L.col_start[n]    = Lnz;
  Urow.row_start[n] = Unz;

  // Fix row inidices of L for final pinv
  for (i_t p = 0; p < Lnz; ++p) {
    L.i[p] = pinv[L.i[p]];
  }

#ifdef CHECK_LOWER_TRIANGULAR
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = L.col_start[j];
    const i_t col_end   = L.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      const i_t i = L.i[p];
      if (i < j) { printf("Found L(%d, %d) not lower triangular!\n", i, j); }
      assert(i >= j);
    }
  }
#endif

  csc_matrix_t<i_t, f_t> U_unpermuted(n, n, 1);
  Urow.to_compressed_col(
    U_unpermuted);  // Convert Urow to U stored in compressed sparse column format
  std::vector<i_t> row_perm(n);
  inverse_permutation(pinv, row_perm);

  std::vector<i_t> identity(n);
  for (i_t k = 0; k < n; k++) {
    identity[k] = k;
  }

  U_unpermuted.permute_rows_and_cols(identity, q, U);

#ifdef CHECK_UPPER_TRIANGULAR
  for (i_t k = 0; k < n; ++k) {
    const i_t j         = k;
    const i_t col_start = U.col_start[j];
    const i_t col_end   = U.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      const i_t i = U.i[p];
      if (i > j) { printf("Found U(%d, %d) not upper triangluar\n", i, j); }
      assert(i <= j);
    }
  }
#endif

  return n;
}

template <typename i_t, typename f_t>
i_t right_looking_lu_row_permutation_only(const csc_matrix_t<i_t, f_t>& A,
                                          const simplex_solver_settings_t<i_t, f_t>& settings,
                                          f_t tol,
                                          f_t start_time,
                                          std::vector<i_t>& q,
                                          std::vector<i_t>& pinv)
{
  // Factorize PAQ = LU, where A is m x n with m >= n, and P and Q are permutation matrices
  // We return the inverser row permutation vector pinv and the column permutation vector q

  f_t factorization_start_time = tic();
  const i_t n                  = A.n;
  const i_t m                  = A.m;
  assert(pinv.size() == m);

  std::vector<i_t> Rdegree(m);  // Rdegree[i] is the degree of row i
  std::vector<i_t> Cdegree(n);  // Cdegree[j] is the degree of column j

  std::vector<std::vector<i_t>> col_count(
    m + 1);  // col_count[nz] is a list of columns with nz nonzeros in the active submatrix
  std::vector<std::vector<i_t>> row_count(
    n + 1);  // row_count[nz] is a list of rows with nz nonzeros in the active submatrix

  std::vector<i_t> column_list(n);
  for (i_t k = 0; k < n; ++k) {
    column_list[k] = k;
  }

  const i_t Bnz = initialize_degree_data(A, column_list, Cdegree, Rdegree, col_count, row_count);
  std::vector<element_t<i_t, f_t>> elements(Bnz);
  std::vector<i_t> first_in_row(m, kNone);
  std::vector<i_t> first_in_col(n, kNone);
  load_elements(A, column_list, Bnz, elements, first_in_row, first_in_col);

  std::vector<i_t> column_j_workspace(m, kNone);
  std::vector<i_t> row_last_workspace(m);
  std::vector<f_t> max_in_column(n);
  std::vector<f_t> max_in_row(m);
  initialize_max_in_column(first_in_col, elements, max_in_column);
#ifdef THRESHOLD_ROOK_PIVOTING
  initialize_max_in_row(first_in_row, elements, max_in_row);
#endif

  settings.log.debug("Empty rows %ld\n", row_count[0].size());
  settings.log.debug("Empty cols %ld\n", col_count[0].size());
  settings.log.debug("Row singletons %ld\n", row_count[1].size());
  settings.log.debug("Col singletons %ld\n", col_count[1].size());

  std::fill(q.begin(), q.end(), -1);
  std::fill(pinv.begin(), pinv.end(), -1);
  std::vector<i_t> qinv(n);
  std::fill(qinv.begin(), qinv.end(), -1);

  f_t last_print = start_time;
  i_t pivots     = 0;
  for (i_t k = 0; k < std::min(m, n); ++k) {
    // Find pivot that satisfies
    // abs(pivot) >= abstol,
    // abs(pivot) >= threshold_tol * max abs[pivot column]
    i_t pivot_i                 = -1;
    i_t pivot_j                 = -1;
    i_t pivot_p                 = kNone;
    constexpr f_t pivot_tol     = 1e-9;
    constexpr f_t drop_tol      = 1e-14;
    constexpr f_t threshold_tol = 1.0 / 10.0;
    // f_t search_start = tic();
    markowitz_search(Cdegree,
                     Rdegree,
                     col_count,
                     row_count,
                     first_in_row,
                     first_in_col,
                     max_in_column,
                     max_in_row,
                     elements,
                     pivot_tol,
                     threshold_tol,
                     pivot_i,
                     pivot_j,
                     pivot_p);
    if (pivot_i == -1 || pivot_j == -1) {
      settings.log.debug("Breaking can't find a pivot %d\n", k);
      break;
    }
    element_t<i_t, f_t>* pivot_entry = &elements[pivot_p];
    assert(pivot_i != -1 && pivot_j != -1);
    assert(pivot_entry->i == pivot_i && pivot_entry->j == pivot_j);

    // Pivot
    pinv[pivot_i]       = k;  // pivot_i is the kth pivot row
    q[k]                = pivot_j;
    qinv[pivot_j]       = k;
    const f_t pivot_val = pivot_entry->x;
    assert(std::abs(pivot_val) >= pivot_tol);
    pivots++;

    // Update Cdegree and col_count
    update_Cdegree_and_col_count<i_t, f_t>(
      pivot_i, pivot_j, first_in_row, Cdegree, col_count, elements);
    update_Rdegree_and_row_count<i_t, f_t>(
      pivot_i, pivot_j, first_in_col, Rdegree, row_count, elements);

    // A22 <- A22 - l u^T
    schur_complement<i_t, f_t>(pivot_i,
                               pivot_j,
                               drop_tol,
                               pivot_val,
                               pivot_p,
                               pivot_entry,
                               first_in_col,
                               first_in_row,
                               row_last_workspace,
                               column_j_workspace,
                               max_in_column,
                               max_in_row,
                               Rdegree,
                               Cdegree,
                               row_count,
                               col_count,
                               elements);

    // Remove the pivot row
    remove_pivot_row<i_t, f_t>(
      pivot_i, pivot_j, first_in_col, first_in_row, max_in_column, elements);
    remove_pivot_col<i_t, f_t>(pivot_i, pivot_j, first_in_col, first_in_row, max_in_row, elements);

    // Set pivot entry to sentinel value
    pivot_entry->i = -1;
    pivot_entry->j = -1;
    pivot_entry->x = std::numeric_limits<f_t>::quiet_NaN();

#ifdef CHECK_MAX_IN_COLUMN
    // Check that maximum in column is maintained
    for (i_t j = 0; j < n; ++j) {
      if (Cdegree[j] == -1) { continue; }
      const f_t max_in_col = max_in_column[j];
      bool found_max       = false;
      f_t largest_abs_x    = 0;
      for (i_t p = first_in_col[j]; p != kNone; p = elements[p].next_in_column) {
        const f_t abs_e2x = std::abs(elements[p].x);
        if (abs_e2x > largest_abs_x) { largest_abs_x = abs_e2x; }
        if (abs_e2x > max_in_col) {
          printf("Found max in column %d is %e but %e\n", j, max_in_col, abs_e2x);
        }
        assert(abs_e2x <= max_in_col);
        if (abs_e2x == max_in_col) { found_max = true; }
      }
      if (!found_max) {
        printf(
          "Did not find max %e in column %d. Largest abs x is %e\n", max_in_col, j, largest_abs_x);
      }
      assert(found_max);
    }
#endif

#if CHECK_BAD_ENTRIES
    for (Int j = 0; j < n; j++) {
      for (Int p = first_in_col[j]; p != kNone; p = elements[p].next_in_column) {
        element_t* entry = &elements[p];
        if (entry->i == -1) { printf("Found bad entry in row %d and column %d\n", entry->i, j); }
        assert(entry->i != -1);
        assert(entry->i != pivot_i);
        assert(entry->j != -1);
        assert(entry->j == j);
        assert(entry->j != pivot_j);
        assert(entry->x == entry->x);
      }
    }

    for (Int i = 0; i < n; i++) {
      for (Int p = first_in_row[i]; p != kNone; p = elements[p].next_in_row) {
        element_t* entry = &elements[p];
        if (entry->i == -1) {
          printf("Bad entry found in row %d. i %d j %d val %e\n", i, entry->i, entry->j, entry->x);
        }
        assert(entry->i != -1);
        assert(entry->i == i);
        assert(entry->i != pivot_i);
        assert(entry->j != -1);
        assert(entry->j != pivot_j);
        assert(entry->x == entry->x);
      }
    }
#endif

    if (toc(last_print) > 10.0) {
      settings.log.printf(
        "Right-looking LU factorization: Pivots %d m %d n %d nelems %ld in "
        "%.2f seconds\n",
        pivots,
        m,
        n,
        elements.size(),
        toc(factorization_start_time));
      last_print = tic();
    }
    if (toc(factorization_start_time) > settings.time_limit) {
      settings.log.printf("Right-looking LU factorization time exceeded\n");
      return -1;
    }

    if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) {
      settings.log.printf("Concurrent halt\n");
      return -2;
    }
  }

  // Finalize the permutation pinv
  // We will have only defined pinv[0..n-1]. When n < m, we still need to define
  // pinv[n..m]
  settings.log.debug("Pivots %d m %d n %d\n", pivots, m, n);
  if (m > n || pivots < n) {
    i_t start = pivots;
    for (i_t i = 0; i < m; ++i) {
      if (pinv[i] == -1) { pinv[i] = start++; }
    }

    // Finalize the permutation q. Do this by first completing the inverse permutation qinv.
    // Then invert qinv to get the final permutation q.
    start = pivots;
    for (i_t j = 0; j < n; ++j) {
      if (qinv[j] == -1) { qinv[j] = start++; }
    }
    inverse_permutation(qinv, q);
  }

  return pivots;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template int right_looking_lu<int, double>(const csc_matrix_t<int, double>& A,
                                           const simplex_solver_settings_t<int, double>& settings,
                                           double tol,
                                           const std::vector<int>& column_list,
                                           std::vector<int>& q,
                                           csc_matrix_t<int, double>& L,
                                           csc_matrix_t<int, double>& U,
                                           std::vector<int>& pinv);

template int right_looking_lu_row_permutation_only<int, double>(
  const csc_matrix_t<int, double>& A,
  const simplex_solver_settings_t<int, double>& settings,
  double tol,
  double start_time,
  std::vector<int>& q,
  std::vector<int>& pinv);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
