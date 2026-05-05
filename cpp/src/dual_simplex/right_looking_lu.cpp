/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/right_looking_lu.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <utilities/memory_instrumentation.hpp>

#include <raft/core/nvtx.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>

using cuopt::ins_vector;

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
};  // 24 bytes
constexpr int kNone = -1;

template <typename i_t, typename f_t>
class nonzero_counts_t {
 public:
  nonzero_counts_t(const std::vector<i_t>& deg, i_t m)
    : m_(m), work_estimate_(0), deg_(deg), counts_(m + 1), pos_(deg.size())
  {
    const i_t n = deg_.size();
    for (i_t k = 0; k < n; ++k) {
      assert(deg_[k] <= m && deg_[k] >= 0);
      const i_t nz = deg_[k];
      pos_[k]      = counts_[nz].size();
      counts_[nz].push_back(k);
    }
    work_estimate_ += 4*n;
  }

  i_t get_count(i_t k) const
  {
    return deg_[k];
  }

  void update_count(i_t k, i_t new_nz)
  {
    const i_t old_nz = deg_[k];
    update_count(k, old_nz, new_nz);
  }

  const std::vector<i_t>& get_elements_with_count(i_t nz) const
  {
    return counts_[nz];
  }

  // Remove k from its current bucket without re-inserting.
  // Sets deg_[k] to -1 to mark it as removed.
  void remove_from_count(i_t k)
  {
    const i_t old_nz = deg_[k];
    const i_t p = pos_[k];
    const i_t other = counts_[old_nz].back();
    counts_[old_nz][p] = other;
    pos_[other] = p;
    counts_[old_nz].pop_back();
    deg_[k] = -1;
    work_estimate_ += 6;
  }

  f_t record_and_clear_work_estimate_() {
    f_t tmp = work_estimate_;
    work_estimate_ = 0;
    return tmp;
  }

 private:

  void update_count(i_t k, i_t old_nz, i_t new_nz)
  {
    const i_t p = pos_[k];
    const i_t other = counts_[old_nz].back();
    counts_[old_nz][p] = other;
    pos_[other] = p;
    counts_[old_nz].pop_back();
    deg_[k] = new_nz;
    pos_[k] = counts_[new_nz].size();
    counts_[new_nz].push_back(k);
    work_estimate_ += 11;
  }

  i_t m_;
  f_t work_estimate_;
  std::vector<i_t> deg_;
  std::vector<std::vector<i_t>> counts_;
  std::vector<i_t> pos_;
};

// Represents the sparse trailing matrix Atlide = A - l u^T of a sparse LU factorization
// We need to be able to access the nonzeros in this matrix by both row and column.
// Thus, we do not compress the storage.
template <typename i_t, typename f_t>
class trailing_matrix_t {
 public:
  trailing_matrix_t(const csc_matrix_t<i_t, f_t>& A,
                    const std::vector<i_t>& column_list)
    : m_(A.m),
      n_(column_list.size()),
      Bnz_(0),
      work_estimate_(0),
      col_start_(n_),
      col_end_(n_),
      col_max_(n_),
      row_start_(m_),
      row_end_(m_),
      row_max_(m_),
      max_in_column_(n_),
      pivot_row_val_(n_, 0.0),
      pivot_col_val_(m_, 0.0),
      pivot_col_mark_(m_, 0),
      row_mark_(m_, kNone),
      col_mark_(n_, kNone),
      col_counts_(compute_column_degree(A, column_list), m_),
      row_counts_(compute_row_degree(A, column_list, Bnz_), n_),
      unused_col_nz_(0),
      unused_row_nz_(0),
      col_hits_(0),
      col_miss_(0),
      row_hits_(0),
      row_miss_(0),
      col_realloc_hist_(std::max(m_, static_cast<i_t>(n_)) + 1, 0),
      row_realloc_hist_(std::max(m_, static_cast<i_t>(n_)) + 1, 0)
  {

    work_estimate_ += 4*m_ + 2*n_ + col_realloc_hist_.size() + row_realloc_hist_.size();

    // Allocate 2x initial size per column/row to reduce early relocations
    i_t col_nz = 2 * Bnz_;
    i_t row_nz = 2 * Bnz_;

    c_i_.resize(col_nz);
    c_x_.resize(col_nz);
    r_j_.resize(row_nz);


    i_t nz = 0;
    for (i_t i = 0; i < m_; i++) {
      row_start_[i] = nz;
      row_end_[i] = nz;  // Temporary value used for initializing r_j_. Will be updated in loop
      i_t row_space = 2 * row_counts_.get_count(i);
      row_max_[i] = nz + row_space;
      nz += row_space;
    }
    assert(nz == row_nz);
    work_estimate_ += 4 * m_;

    nz = 0;
    for (size_t k = 0; k < column_list.size(); k++) {
      const i_t j = column_list[k];
      const i_t A_start = A.col_start[j];
      const i_t A_end = A.col_start[j + 1];
      const i_t len  = A_end - A_start;
      i_t col_space = 2 * len;
      col_max_[k] = nz + col_space;
      col_start_[k] = nz;
      col_end_[k] = nz + len;
      for (i_t p = A_start; p < A_end; p++) {
        const i_t row = A.i[p];
        const f_t val = A.x[p];
        c_i_[nz] = row;
        c_x_[nz] = val;
        nz++;
        r_j_[row_end_[row]] = k;
        row_end_[row]++;
      }
      nz += col_space - len; // Remaining capacity for this column
    }
    assert(nz == col_nz);
    work_estimate_ += 7 * n_ + 7 * Bnz_;

    for (i_t j = 0; j < n_; j++) {
      f_t max_in_col = 0.0;
      const i_t c_start = col_start_[j];
      const i_t c_end = col_end_[j];
      for (i_t p = c_start; p < c_end; p++) {
        const f_t val = std::abs(c_x_[p]);
        if (val > max_in_col) {
          max_in_col = val;
        }
      }
      max_in_column_[j] = max_in_col;
    }
    work_estimate_ += Bnz_ + 3*n_;
  }

  f_t record_and_clear_work_estimate_()
  {
    const f_t row_work_estimate = row_counts_.record_and_clear_work_estimate_();
    const f_t col_work_estimate = col_counts_.record_and_clear_work_estimate_();
    work_estimate_ += row_work_estimate + col_work_estimate;
    f_t tmp        = work_estimate_;
    work_estimate_ = 0;
    return tmp;
  }

  i_t markowitz_search(f_t pivot_tol, f_t threshold_tol, i_t& pivot_i, i_t& pivot_j, f_t &pivot_val) {
    f_t markowitz = static_cast<f_t>(m_) * static_cast<f_t>(n_); // Upper bound on markowitz criteria
    i_t nz      = 1;
    i_t nsearch            = 0;
    constexpr bool verbose = false;
    i_t nz_max             = std::min(m_, n_);
    while (nz <= nz_max) {
      i_t markowitz_lower_bound = (nz - 1) * (nz - 1);
      // Search columns of length nz
      i_t nsearch_start = nsearch;
      for (const i_t j : col_counts_.get_elements_with_count(nz)) {
        assert(col_counts_.get_count(j) == nz);
        const f_t max_in_col = max_in_column_[j];
        const i_t c_start = col_start_[j];
        const i_t c_end = col_end_[j];
        for (i_t p = c_start; p < c_end; p++) {
          const i_t i = c_i_[p];
          const f_t val = c_x_[p];
          const i_t rdeg = row_counts_.get_count(i);
          assert(rdeg >= 0);
          const i_t Mij = (rdeg - 1) * (nz - 1);
          if (Mij < markowitz && std::abs(val) >= threshold_tol * max_in_col &&
              std::abs(val) >= pivot_tol) {
            markowitz = Mij;
            pivot_i   = i;
            pivot_j   = j;
            pivot_val = val;
            if (markowitz <= markowitz_lower_bound) { break; }
          }
        }
        work_estimate_ += 3 * (c_end - c_start);
        nsearch++;
        if (markowitz <= markowitz_lower_bound) { break; }
      }
     work_estimate_ += 4 * (nsearch - nsearch_start);
      if (markowitz <= markowitz_lower_bound) { break; }

      markowitz_lower_bound = (nz - 1) * nz;

      // Search rows of length nz
      assert(row_counts_.get_elements_with_count(nz).size() >= 0);
      nsearch_start = nsearch;
      for (const i_t i : row_counts_.get_elements_with_count(nz)) {
        const i_t rdeg = row_counts_.get_count(i);
        assert(rdeg == nz);
        const i_t r_start = row_start_[i];
        const i_t r_end = row_end_[i];
        for (i_t p = r_start; p < r_end; p++) {
          const i_t j = r_j_[p];
          // Look up the value from the column copy of j
          f_t val = 0;
          const i_t c_start = col_start_[j];
          const i_t c_end = col_end_[j];
          for (i_t q = c_start; q < c_end; q++) {
            if (c_i_[q] == i) { val = c_x_[q]; break; }
          }
          work_estimate_ += 2 * (c_end - c_start);
          const f_t max_in_col = max_in_column_[j];
          const i_t cdeg = col_counts_.get_count(j);
          assert(cdeg >= 0);
          const i_t Mij = (nz - 1) * (cdeg - 1);
          if (Mij < markowitz && std::abs(val) >= threshold_tol * max_in_col &&
              std::abs(val) >= pivot_tol) {
            markowitz = Mij;
            pivot_i   = i;
            pivot_j   = j;
            pivot_val = val;
            if (markowitz <= markowitz_lower_bound) { break; }
          }
        }
        work_estimate_ += 5 * (r_end - r_start);
        nsearch++;
        if (markowitz <= markowitz_lower_bound) { break; }
      }
      work_estimate_ += 4 * (nsearch - nsearch_start);
      if (pivot_i != -1 && nz >= 2) { break; }
      nz++;
    }
    if (nsearch > 10) {
      if constexpr (verbose) { printf("nsearch %d\n", nsearch); }
    }
    return nsearch;
  }


  void update_for_pivot_removal(i_t pivot_i, i_t pivot_j)
  {
    // Iterate over the pivot row: decrement column degrees.
    // Skip the pivot column itself — it is being eliminated, not just decremented.
    const i_t r_start = row_start_[pivot_i];
    const i_t r_end = row_end_[pivot_i];
    for (i_t p = r_start; p < r_end; p++) {
      const i_t j = r_j_[p];
      const i_t cdeg = col_counts_.get_count(j);
      if (j != pivot_j) {
        col_counts_.update_count(j, cdeg - 1);
      } else {
        col_counts_.remove_from_count(j);
      }
    }
    work_estimate_ += 2 * (r_end - r_start);

    // Iterate over the pivot column: decrement row degrees.
    // Skip the pivot row itself — it is being eliminated, not just decremented.
    const i_t c_start = col_start_[pivot_j];
    const i_t c_end = col_end_[pivot_j];
    for (i_t p = c_start; p < c_end; p++) {
      const i_t i = c_i_[p];
      const i_t rdeg = row_counts_.get_count(i);
      if (i != pivot_i) {
        row_counts_.update_count(i, rdeg - 1);
      } else {
        row_counts_.remove_from_count(i);
      }
    }
    work_estimate_ += 2 * (c_end - c_start);
  }

  void schur_complement(i_t pivot_i,
                        i_t pivot_j,
                        f_t drop_tol,
                        f_t pivot_val)
  {
    // Step 1: Cache the pivot column into dense workspaces.
    // pivot_col_val_[i] = l_i = a(i, pivot_j) / pivot_val  for each row i != pivot_i
    // pivot_col_mark_[i] = 1 if row i is in the pivot column
    // pivot_col_index_[] = sparse list of such row indices
    i_t pivot_col_count = 0;
    const i_t c_pivot_start = col_start_[pivot_j];
    const i_t c_pivot_end = col_end_[pivot_j];
    for (i_t p = c_pivot_start; p < c_pivot_end; p++) {
      const i_t i = c_i_[p];
      if (i == pivot_i) { continue; }
      const f_t li = c_x_[p] / pivot_val;
      pivot_col_val_[i] = li;
      pivot_col_mark_[i] = 1;
      pivot_col_index_.push_back(i);
      pivot_col_count++;
    }
    work_estimate_ += 5 * (c_pivot_end - c_pivot_start);

    // Step 2: For each column j in the pivot row, update existing entries and insert fill.
    const i_t r_pivot_start = row_start_[pivot_i];
    const i_t r_pivot_end = row_end_[pivot_i];
    for (i_t p0 = r_pivot_start; p0 < r_pivot_end; p0++) {
      const i_t j = r_j_[p0];
      if (j == pivot_j) { continue; }
      const f_t uj = pivot_row_val_[j];

      // Step 2a: Scan column j, update existing entries, and count fill-in.
      // For each entry (i, j) that also appears in the pivot column, update it.
      // Simultaneously, unmark pivot_col_mark_[i] for matched entries, so that
      // after the scan, the still-marked entries are the fill-ins.
      i_t n_fillin = pivot_col_count;
      i_t n_cancel = 0;
      const i_t c_start = col_start_[j];
      const i_t c_end = col_end_[j];
      for (i_t q = c_start; q < c_end; q++) {
        const i_t i = c_i_[q];
        if (pivot_col_mark_[i]) {
          pivot_col_mark_[i] = 0;
          n_fillin--;
          const f_t val = pivot_col_val_[i] * uj;

          c_x_[q] -= val;
          const f_t abs_updated = std::abs(c_x_[q]);
          if (abs_updated > max_in_column_[j]) { max_in_column_[j] = abs_updated; }
          if (abs_updated < drop_tol) {
            c_x_[q] = 0;
            n_cancel++;
            // TODO: does max_in_column_ need to be updated in this case?
          }
        }
      }
      work_estimate_ += 2*(c_end - c_start) + 6*(pivot_col_count - n_fillin);


      // Step 2b: Remove cancellations (entries that became zero).
      if (n_cancel > 0) {
        i_t new_end = col_start_[j];
        for (i_t q = col_start_[j]; q < col_end_[j]; q++) {
          if (c_x_[q] != 0) {
            c_i_[new_end] = c_i_[q];
            c_x_[new_end] = c_x_[q];
            new_end++;
          } else {
            const i_t dead_row = c_i_[q];
            // Remove this entry from the row copy as well
            const i_t r_start = row_start_[dead_row];
            for (i_t rp = r_start; rp < row_end_[dead_row]; rp++) {
              if (r_j_[rp] == j) {
                r_j_[rp] = r_j_[row_end_[dead_row] - 1];
                row_end_[dead_row]--;
                break;
              }
            }
            work_estimate_ += 2*(row_end_[dead_row] - r_start) + 4;
            // Update row degree
            const i_t rdeg = row_counts_.get_count(dead_row);
            row_counts_.update_count(dead_row, rdeg - 1);

          }
        }
        work_estimate_ += 2*(new_end - col_start_[j]);
        i_t old_count = col_end_[j] - col_start_[j];
        col_end_[j] = new_end;
        i_t new_count = new_end - col_start_[j];
        // Update column degree for cancellations
        if (new_count != old_count) {
          col_counts_.update_count(j, new_count);
        }
      }


      // Step 2c: Insert fill-in entries. We know exactly how many there are.
      if (n_fillin > 0) {
        // Ensure column j has enough space for all fill-ins at once.
        // After this, col_start_[j] is stable — no further relocation needed.
        ensure_col_space(j, n_fillin);

        // Insert fill into column j and row copies.
        for (i_t k = 0; k < pivot_col_count; k++) {
          const i_t i = pivot_col_index_[k];
          if (pivot_col_mark_[i]) {
            const f_t val = pivot_col_val_[i] * uj;
            const f_t abs_val = std::abs(val);
            if (abs_val < drop_tol) {
              // Skip this fill-in but still need to unmark
              continue;
            }

            // Insert into column copy (space is guaranteed)
            c_i_[col_end_[j]] = i;
            c_x_[col_end_[j]] = -val;
            col_end_[j]++;
            if (abs_val > max_in_column_[j]) { max_in_column_[j] = abs_val; }

            // Insert into row copy
            ensure_row_space(i, 1);
            r_j_[row_end_[i]] = j;
            row_end_[i]++;

            // Update row degree
            const i_t rdeg = row_counts_.get_count(i);
            row_counts_.update_count(i, rdeg + 1);
            work_estimate_ += 10;
          }
        }
      }

      // Step 2d: Update column degree bucket once for this column.
      {
        i_t new_cdeg = col_end_[j] - col_start_[j];
        if (new_cdeg != col_counts_.get_count(j)) {
          col_counts_.update_count(j, new_cdeg);
        }
      }

      // Step 2e: Reset all pivot column marks back to 1 for the next column.
      // Some marks were cleared to 0 during the scan of column j (matched entries).
      // We restore them by iterating the pivot column index list. So that we are
      // prepared to process the next column.
      for (i_t k = 0; k < pivot_col_count; k++) {
        pivot_col_mark_[pivot_col_index_[k]] = 1;
      }
      work_estimate_ += 2*pivot_col_count;
    }

    // Step 3: Clear the pivot column workspaces.
    for (i_t k = 0; k < pivot_col_count; k++) {
      const i_t i = pivot_col_index_[k];
      pivot_col_val_[i] = 0;
      pivot_col_mark_[i] = 0;
    }
    work_estimate_ += 2*pivot_col_count;
    pivot_col_index_.clear();
  }

  // Populate the dense pivot_row_val_ workspace by scanning column representation
  // for each column j that appears in the pivot row.
  // Must be called before extract_row() and schur_complement().
  // Cleared by remove_pivot_row_and_column().
  void cache_pivot_row(i_t pivot_i)
  {
    const i_t r_start = row_start_[pivot_i];
    const i_t r_end = row_end_[pivot_i];
    for (i_t p = r_start; p < r_end; p++) {
      const i_t j = r_j_[p];
      const i_t c_start = col_start_[j];
      const i_t c_end = col_end_[j];
      i_t q;
      for (q = c_start; q < c_end; q++) {
        if (c_i_[q] == pivot_i) {
          pivot_row_val_[j] = c_x_[q];
          break;
        }
      }
      work_estimate_ += 2 *(q - c_start);
    }
    work_estimate_ += 3 * (r_end - r_start) + 2;
  }

  void remove_pivot_row_and_column(i_t pivot_i, i_t pivot_j)
  {
    // Iterate over the pivot row
    const i_t r_pivot_start = row_start_[pivot_i];
    const i_t r_pivot_end = row_end_[pivot_i];
    for (i_t p = r_pivot_start; p < r_pivot_end; p++) {
      const i_t j = r_j_[p];
      // Clear the cached pivot row value for this column
      pivot_row_val_[j] = 0;
      // Remove pivot_i from each column j in the pivot row
      f_t max_in_col = 0.0;

      const i_t prev_col_end = col_end_[j];
      for (i_t q = col_start_[j]; q < col_end_[j]; q++) {
        const i_t i = c_i_[q];
        if (i == pivot_i) {
          // Swap with the last element in the column
          i_t other_i = c_i_[col_end_[j] - 1];
          f_t other_x = c_x_[col_end_[j] - 1];
          c_i_[q] = other_i;
          c_x_[q] = other_x;
          // Update col_end_[j]
          col_end_[j]--;
          q--;
          continue;
        } else {
          const f_t val = std::abs(c_x_[q]);
          if (val > max_in_col) {
            max_in_col = val;
          }
        }
      }
      work_estimate_ += 3*(prev_col_end - col_start_[j]) + 7;
      max_in_column_[j] = max_in_col;
    }
    work_estimate_ += 4*(r_pivot_end - r_pivot_start);


    // Iterate over the pivot column
    const i_t c_start = col_start_[pivot_j];
    const i_t c_end = col_end_[pivot_j];
    for (i_t p = c_start; p < c_end; p++) {
      const i_t i = c_i_[p];
      // Remove pivot_j from each row i in the pivot column

      i_t q;
      for (q = row_start_[i]; q < row_end_[i]; q++) {
        const i_t j = r_j_[q];
        if (j == pivot_j) {
          // Swap with the last element in the row
          r_j_[q] = r_j_[row_end_[i] - 1];
          // Update row_end_[i]
          row_end_[i]--;
          break;
        }
      }
      work_estimate_ += 2*(q - row_start_[i]) + 4;
    }
    work_estimate_ += 4*(c_end - c_start);

    // Mark pivot column and pivot row as empty so garbage collection skips them
    col_end_[pivot_j] = col_start_[pivot_j];
    row_end_[pivot_i] = row_start_[pivot_i];
  }

  void extract_row(i_t pivot_i, i_t pivot_j, csr_matrix_t<i_t, f_t>& Urow, i_t& Unz)
  {
    // U(k, :)
    const i_t r_pivot_start = row_start_[pivot_i];
    const i_t r_pivot_end = row_end_[pivot_i];
    for (i_t p = r_pivot_start; p < r_pivot_end; p++) {
      const i_t j = r_j_[p];
      if (j != pivot_j) {
        Urow.j.push_back(j);
        Urow.x.push_back(pivot_row_val_[j]);
        Unz++;
      }
    }
    work_estimate_ += 3 * (r_pivot_end - r_pivot_start);
  }

  void extract_column(i_t pivot_i, i_t pivot_j, f_t pivot_val, csc_matrix_t<i_t, f_t>& L, i_t& Lnz)
  {
    // L(:, k)
    const i_t c_pivot_start = col_start_[pivot_j];
    const i_t c_pivot_end = col_end_[pivot_j];
    for (i_t p = c_pivot_start; p < c_pivot_end; p++) {
      const i_t i = c_i_[p];
      if (i != pivot_i) {
        L.i.push_back(i);
        const f_t l_val = c_x_[p] / pivot_val;
        L.x.push_back(l_val);
        Lnz++;
      }
    }
    work_estimate_ += 4 * (c_pivot_end - c_pivot_start);
  }

  void garbage_collect(f_t max_unused_fraction = 0.90)
  {
    if (unused_col_nz_ > max_unused_fraction * static_cast<f_t>(c_i_.size())) {
      printf("Garbage collected column %e\n", unused_col_nz_ / static_cast<f_t>(c_i_.size()));
      std::vector<i_t> new_c_i;
      std::vector<f_t> new_c_x;
      new_c_i.reserve(c_i_.size() - unused_col_nz_);
      new_c_x.reserve(c_x_.size() - unused_col_nz_);
      for (i_t j = 0; j < n_; j++) {
        const i_t new_start = static_cast<i_t>(new_c_i.size());
        const i_t c_start = col_start_[j];
        const i_t c_end = col_end_[j];
        const i_t col_size = c_end - c_start;
        for (i_t p = c_start; p < c_end; p++) {
          new_c_i.push_back(c_i_[p]);
          new_c_x.push_back(c_x_[p]);
        }
        col_start_[j] = new_start;
        col_end_[j] = static_cast<i_t>(new_c_i.size());
        // Reserve space equal to current size (doubling strategy)
        for (i_t s = 0; s < col_size; s++) {
          new_c_i.push_back(kNone);
          new_c_x.push_back(0.0);
        }
        work_estimate_ += 4*col_size;
        col_max_[j] = static_cast<i_t>(new_c_i.size());
      }
      work_estimate_ += 6*n_;
      c_i_ = std::move(new_c_i);
      c_x_ = std::move(new_c_x);

      unused_col_nz_ = 0;
    }

    if (unused_row_nz_ > max_unused_fraction * static_cast<f_t>(r_j_.size())) {
      printf("Garbage collected row %e\n", unused_row_nz_ / static_cast<f_t>(r_j_.size()));
      std::vector<i_t> new_r_j;
      new_r_j.reserve(r_j_.size() - unused_row_nz_);
      for (i_t i = 0; i < m_; i++) {
        const i_t new_start = static_cast<i_t>(new_r_j.size());
        const i_t r_start = row_start_[i];
        const i_t r_end = row_end_[i];
        const i_t row_size = r_end - r_start;
        for (i_t p = r_start; p < r_end; p++) {
          new_r_j.push_back(r_j_[p]);
        }
        row_start_[i] = new_start;
        row_end_[i] = static_cast<i_t>(new_r_j.size());
        // Reserve space equal to current size (doubling strategy)
        for (i_t s = 0; s < row_size; s++) {
          new_r_j.push_back(kNone);
        }
        row_max_[i] = static_cast<i_t>(new_r_j.size());
        work_estimate_ += 2*row_size;
      }
      work_estimate_ += 6*m_;
      r_j_ = std::move(new_r_j);
      unused_row_nz_ = 0;
    }
  }

  void print_stats()
  {
#if 0
    printf("Column hits: %.1f%%, Column misses: %.1f%%, Row hits: %.1f%%, Row misses: %.1f%%\n",
           100.0 * static_cast<f_t>(col_hits_) / static_cast<f_t>(col_hits_ + col_miss_),
           100.0 * static_cast<f_t>(col_miss_) / static_cast<f_t>(col_hits_ + col_miss_),
           100.0 * static_cast<f_t>(row_hits_) / static_cast<f_t>(row_hits_ + row_miss_),
           100.0 * static_cast<f_t>(row_miss_) / static_cast<f_t>(row_hits_ + row_miss_));

    printf("Column reallocation histogram (shortfall -> count):\n");
    for (size_t k = 0; k < col_realloc_hist_.size(); k++) {
      if (col_realloc_hist_[k] > 0) {
        printf("  %4zu: %d\n", k, col_realloc_hist_[k]);
      }
    }

    printf("Row reallocation histogram (shortfall -> count):\n");
    for (size_t k = 0; k < row_realloc_hist_.size(); k++) {
      if (row_realloc_hist_[k] > 0) {
        printf("  %4zu: %d\n", k, row_realloc_hist_[k]);
      }
    }

    f_t ci_mb = static_cast<f_t>(c_i_.size() * sizeof(i_t)) / (1024.0 * 1024.0);
    f_t cx_mb = static_cast<f_t>(c_x_.size() * sizeof(f_t)) / (1024.0 * 1024.0);
    f_t rj_mb = static_cast<f_t>(r_j_.size() * sizeof(i_t)) / (1024.0 * 1024.0);
    printf("Memory: c_i_ = %.2f MB, c_x_ = %.2f MB, r_j_ = %.2f MB, total = %.2f MB\n",
           ci_mb, cx_mb, rj_mb, ci_mb + cx_mb + rj_mb);
#endif
  }

 private:

  // Ensure column j has space for at least `needed` additional entries.
  // If not, relocate the column to the end of c_i_/c_x_ with enough space.
  // Returns true if the column was relocated (invalidating any cached positions).
  bool ensure_col_space(i_t j, i_t needed)
  {
    if (col_end_[j] + needed <= col_max_[j]) {
      col_hits_++;
      return false;
    }
    col_miss_++;
    i_t shortfall = needed - (col_max_[j] - col_end_[j]);
    col_realloc_hist_[shortfall]++;
    // Relocate column j to the end of c_i_/c_x_
    const i_t c_start = col_start_[j];
    const i_t c_end = col_end_[j];
    i_t current_size = c_end - c_start;
    unused_col_nz_ += current_size;
    i_t new_start = c_i_.size();
    for (i_t p = c_start; p < c_end; p++) {
      c_i_.push_back(c_i_[p]);
      c_x_.push_back(c_x_[p]);
    }
    work_estimate_ += 2*(c_end - c_start);
    col_start_[j] = new_start;
    col_end_[j] = c_i_.size();
    // Reserve space using doubling strategy to reduce future relocations
    i_t extra = std::max(current_size, needed);
    for (i_t k = 0; k < extra; k++) {
      c_i_.push_back(kNone);
      c_x_.push_back(0.0);
    }
    work_estimate_ += 2*extra;
    col_max_[j] = c_i_.size();
    work_estimate_ += 10;
    return true;
  }

  // Ensure row i has space for at least `needed` additional entries.
  // If not, relocate the row to the end of r_j_ with enough space.
  void ensure_row_space(i_t i, i_t needed)
  {
    if (row_end_[i] + needed <= row_max_[i]) {
      row_hits_++;
      return;
    }
    row_miss_++;
    i_t shortfall = needed - (row_max_[i] - row_end_[i]);
    row_realloc_hist_[shortfall]++;
    // Relocate row i to the end of r_j_
    const i_t r_start = row_start_[i];
    const i_t r_end = row_end_[i];
    i_t current_size = r_end - r_start;
    unused_row_nz_ += current_size;
    i_t new_start = r_j_.size();
    for (i_t p = r_start; p < r_end; p++) {
      r_j_.push_back(r_j_[p]);
    }
    work_estimate_ += (r_end - r_start);
    row_start_[i] = new_start;
    row_end_[i] = r_j_.size();
    // Reserve space using doubling strategy to reduce future relocations
    i_t extra = std::max(current_size, needed);
    for (i_t k = 0; k < extra; k++) {
      r_j_.push_back(kNone);
    }
    work_estimate_ += extra;
    row_max_[i] = r_j_.size();
    work_estimate_ += 9;
  }

  std::vector<i_t> compute_column_degree(const csc_matrix_t<i_t, f_t>& A, const std::vector<i_t>& column_list)
  {
    const i_t n = column_list.size();
    std::vector<i_t> Cdegree(n);
    for (i_t k = 0; k < n; k++) {
      const i_t j         = column_list[k];
      const i_t A_start = A.col_start[j];
      const i_t A_end   = A.col_start[j + 1];
      Cdegree[k]          = A_end - A_start;
    }
    work_estimate_ += 4 * n;
    return Cdegree;
  }

  std::vector<i_t> compute_row_degree(const csc_matrix_t<i_t, f_t>& A, const std::vector<i_t>& column_list, i_t& Bnz)
  {
    std::vector<i_t> Rdegree(A.m, 0);
    Bnz = 0;
    const i_t n = column_list.size();
    for (i_t k = 0; k < n; k++) {
      const i_t j = column_list[k];
      const i_t col_start = A.col_start[j];
      const i_t col_end = A.col_start[j + 1];
      for (i_t p = col_start; p < col_end; ++p) {
        Rdegree[A.i[p]]++;
        Bnz++;
      }
    }
    work_estimate_ += 3 * n + 2 * Bnz;
    return Rdegree;
  }

  i_t m_;
  i_t n_;
  i_t Bnz_;
  f_t work_estimate_;

  // The representation of the matrix by column
  std::vector<i_t> col_start_;
  std::vector<i_t> col_end_;
  std::vector<i_t> col_max_;

  std::vector<i_t> c_i_;   // row indices (indexed by col_start_[j] to col_end_[j])
  std::vector<f_t> c_x_; // coefficients (indexed by col_start_[j] to col_end_[j])


  // The representation of the matrix by row (index only, no values)
  std::vector<i_t> row_start_;
  std::vector<i_t> row_end_;
  std::vector<i_t> row_max_;

  std::vector<i_t> r_j_;   // column indices (indexed by row_start_[i] to row_end_[i])



  std::vector<f_t> max_in_column_;  // max_in_column_[j] is absolute value of the maximum coefficient in column j

  std::vector<f_t> pivot_row_val_;  // dense workspace of size n_; caches pivot row values

  std::vector<f_t> pivot_col_val_;   // dense workspace of size m_; caches L multipliers for pivot column
  std::vector<char> pivot_col_mark_; // dense workspace of size m_; 1 if row i is in the pivot column
  std::vector<i_t> pivot_col_index_; // sparse list of row indices in the pivot column (excl. pivot_i)

  std::vector<i_t> row_mark_;
  std::vector<i_t> col_mark_;


  nonzero_counts_t<i_t, f_t> col_counts_;
  nonzero_counts_t<i_t, f_t> row_counts_;


  i_t unused_col_nz_;
  i_t unused_row_nz_;


  i_t col_hits_;
  i_t col_miss_;
  i_t row_hits_;
  i_t row_miss_;

  std::vector<i_t> col_realloc_hist_;  // col_realloc_hist_[k] = number of column relocations with shortfall k
  std::vector<i_t> row_realloc_hist_;  // row_realloc_hist_[k] = number of row relocations with shortfall k
};

template <typename i_t, typename f_t>
i_t initialize_degree_data(const csc_matrix_t<i_t, f_t>& A,
                           const std::vector<i_t>& column_list,
                           std::vector<i_t>& Cdegree,
                           std::vector<i_t>& Rdegree,
                           std::vector<std::vector<i_t>>& col_count,
                           std::vector<std::vector<i_t>>& row_count,
                           f_t& work_estimate)
{
  const i_t n = column_list.size();
  const i_t m = A.m;
  std::fill(Rdegree.begin(), Rdegree.end(), 0);
  work_estimate += Rdegree.size();

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
  work_estimate += 3 * n + 2 * Bnz;

  for (i_t k = 0; k < n; ++k) {
    assert(Cdegree[k] <= m && Cdegree[k] >= 0);
    col_count[Cdegree[k]].push_back(k);
  }
  work_estimate += 3 * n;

  for (i_t k = 0; k < m; ++k) {
    assert(Rdegree[k] <= n && Rdegree[k] >= 0);
    row_count[Rdegree[k]].push_back(k);
    if (Rdegree[k] == 0) {
      constexpr bool verbose = false;
      if (verbose) { printf("Zero degree row %d\n", k); }
    }
  }
  work_estimate += 4 * m;

  return Bnz;
}

// Fill col_pos and row_pos so that column j has col_pos[j] = its index in col_count[Cdegree[j]],
// and row i has row_pos[i] = its index in row_count[Rdegree[i]]. Enables O(1) degree-bucket
// removal.
template <typename i_t>
void initialize_bucket_positions(const std::vector<std::vector<i_t>>& col_count,
                                 const std::vector<std::vector<i_t>>& row_count,
                                 i_t col_max_degree,
                                 i_t row_max_degree,
                                 std::vector<i_t>& col_pos,
                                 std::vector<i_t>& row_pos)
{
  for (i_t d = 0; d <= col_max_degree; ++d) {
    for (i_t pos = 0; pos < static_cast<i_t>(col_count[d].size()); ++pos) {
      col_pos[col_count[d][pos]] = pos;
    }
  }
  for (i_t d = 0; d <= row_max_degree; ++d) {
    for (i_t pos = 0; pos < static_cast<i_t>(row_count[d].size()); ++pos) {
      row_pos[row_count[d][pos]] = pos;
    }
  }
}

template <typename i_t, typename f_t>
i_t load_elements(const csc_matrix_t<i_t, f_t>& A,
                  const std::vector<i_t>& column_list,
                  i_t Bnz,
                  std::vector<element_t<i_t, f_t>>& elements,
                  std::vector<i_t>& first_in_row,
                  std::vector<i_t>& first_in_col,
                  std::vector<i_t>& last_in_row,
                  f_t& work_estimate)
{
  const i_t m = A.m;
  const i_t n = column_list.size();
  work_estimate += m;

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
      // if there will be more entries in this row yet))
      if (last_in_row[i] != kNone) {
        // If we have seen an entry in this row before, set the last entry we've seen in this row to
        // point to the current entry
        elements[last_in_row[i]].next_in_row = nz;
      }
      // The current entry becomes the last element seen in the row
      last_in_row[i] = nz;
      if (p == col_start) { first_in_col[k] = nz; }
      if (first_in_row[i] == kNone) { first_in_row[i] = nz; }
      nz++;
    }
  }
  work_estimate += 3 * n + 15 * nz;
  assert(nz == Bnz);

  for (i_t j = 0; j < n; j++) {
    for (i_t p = first_in_col[j]; p != kNone; p = elements[p].next_in_column) {
      element_t<i_t, f_t>* entry = &elements[p];
      assert(entry->j == j);
      assert(entry->i >= 0);
      assert(entry->i < m);
    }
  }
  work_estimate += 2 * n + nz;

  for (i_t i = 0; i < m; i++) {
    for (i_t p = first_in_row[i]; p != kNone; p = elements[p].next_in_row) {
      element_t<i_t, f_t>* entry = &elements[p];
      assert(entry->i == i);
      assert(entry->j < n);
      assert(entry->j >= 0);
    }
  }
  work_estimate += 2 * m + nz;

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
                     i_t& pivot_p,
                     f_t& work_estimate)
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

      work_estimate += 3 * nz;
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
      work_estimate += 5 * nz;
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
                                  std::vector<i_t>& col_pos,
                                  std::vector<element_t<i_t, f_t>>& elements,
                                  f_t& work_estimate)
{
  // Update Cdegree and col_count (O(1) removal using position array)
  i_t loop_count = 0;
  for (i_t p = first_in_row[pivot_i]; p != kNone; p = elements[p].next_in_row) {
    element_t<i_t, f_t>* entry = &elements[p];
    const i_t j                = entry->j;
    assert(entry->i == pivot_i);
    i_t cdeg = Cdegree[j];
    assert(cdeg >= 0);
    // O(1) swap-with-last removal
    {
      i_t pos              = col_pos[j];
      i_t other            = col_count[cdeg].back();
      col_count[cdeg][pos] = other;
      col_pos[other]       = pos;
      col_count[cdeg].pop_back();
    }
    cdeg = --Cdegree[j];
    assert(cdeg >= 0);
    if (j != pivot_j && cdeg >= 0) {
      col_pos[j] = col_count[cdeg].size();
      col_count[cdeg].push_back(j);
    }
    loop_count++;
  }
  work_estimate += 7 * loop_count;
  Cdegree[pivot_j] = -1;
}

template <typename i_t, typename f_t>
void update_Rdegree_and_row_count(i_t pivot_i,
                                  i_t pivot_j,
                                  const std::vector<i_t>& first_in_col,
                                  std::vector<i_t>& Rdegree,
                                  std::vector<std::vector<i_t>>& row_count,
                                  std::vector<i_t>& row_pos,
                                  std::vector<element_t<i_t, f_t>>& elements,
                                  f_t& work_estimate)
{
  // Update Rdegree and row_count (O(1) removal using position array)
  i_t loop_count = 0;
  for (i_t p = first_in_col[pivot_j]; p != kNone; p = elements[p].next_in_column) {
    element_t<i_t, f_t>* entry = &elements[p];
    const i_t i                = entry->i;
    i_t rdeg                   = Rdegree[i];
    assert(rdeg >= 0);
    // O(1) swap-with-last removal
    {
      i_t pos              = row_pos[i];
      i_t other            = row_count[rdeg].back();
      row_count[rdeg][pos] = other;
      row_pos[other]       = pos;
      row_count[rdeg].pop_back();
    }
    rdeg = --Rdegree[i];
    assert(rdeg >= 0);
    if (i != pivot_i && rdeg >= 0) {
      row_pos[i] = row_count[rdeg].size();
      row_count[rdeg].push_back(i);
    }
    loop_count++;
  }
  work_estimate += 7 * loop_count;
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
                      std::vector<i_t>& last_in_row,
                      std::vector<i_t>& col_pos,
                      std::vector<i_t>& row_pos,
                      std::vector<element_t<i_t, f_t>>& elements,
                      f_t& work_estimate)
{
  // row_last_workspace: temp copy of last_in_row for this pivot step, updated when adding fill
  // last_in_row: persistent tail pointer per row
  for (i_t p1 = first_in_col[pivot_j]; p1 != kNone; p1 = elements[p1].next_in_column) {
    const i_t i           = elements[p1].i;
    row_last_workspace[i] = last_in_row[i];
  }
  work_estimate += 4 * Cdegree[pivot_j];

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
    work_estimate += 5 * Cdegree[j];

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
        last_in_row[i]        = fill_p;  // maintain last_in_row persistent state
        // Row degree update: O(1) removal using row_pos
        {
          i_t rdeg             = Rdegree[i];
          i_t pos              = row_pos[i];
          i_t other            = row_count[rdeg].back();
          row_count[rdeg][pos] = other;
          row_pos[other]       = pos;
          row_count[rdeg].pop_back();
          row_pos[i] = row_count[rdeg + 1].size();
          row_count[++Rdegree[i]].push_back(i);
        }
        // Col degree update: O(1) removal using col_pos
        {
          i_t cdeg             = Cdegree[j];
          i_t pos              = col_pos[j];
          i_t other            = col_count[cdeg].back();
          col_count[cdeg][pos] = other;
          col_pos[other]       = pos;
          col_count[cdeg].pop_back();
          col_pos[j] = col_count[cdeg + 1].size();
          col_count[++Cdegree[j]].push_back(j);
        }
      }
    }
    work_estimate += 10 * Cdegree[pivot_j];

    for (i_t p1 = first_in_col[j]; p1 != kNone; p1 = elements[p1].next_in_column) {
      element_t<i_t, f_t>* e = &elements[p1];
      const i_t i            = e->i;
      assert(e->j == j);
      column_j_workspace[i] = kNone;
    }
    work_estimate += 5 * Cdegree[j];
  }
  work_estimate += 10 * Rdegree[pivot_i];
}

template <typename i_t, typename f_t>
void remove_pivot_row(i_t pivot_i,
                      i_t pivot_j,
                      std::vector<i_t>& first_in_col,
                      std::vector<i_t>& first_in_row,
                      std::vector<f_t>& max_in_column,
                      std::vector<element_t<i_t, f_t>>& elements,
                      f_t& work_estimate)
{
  // Remove the pivot row
  i_t row_loop_count = 0;
  for (i_t p0 = first_in_row[pivot_i]; p0 != kNone; p0 = elements[p0].next_in_row) {
    element_t<i_t, f_t>* e = &elements[p0];
    const i_t j            = e->j;
    if (j == pivot_j) { continue; }
    i_t last           = kNone;
    f_t max_in_col_j   = 0;
    i_t col_loop_count = 0;
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
      col_loop_count++;
    }
    work_estimate += 3 * col_loop_count;
    max_in_column[j] = max_in_col_j;
    row_loop_count++;
  }
  work_estimate += 5 * row_loop_count;

  first_in_row[pivot_i] = kNone;
}

template <typename i_t, typename f_t>
void remove_pivot_col(i_t pivot_i,
                      i_t pivot_j,
                      std::vector<i_t>& first_in_col,
                      std::vector<i_t>& first_in_row,
                      std::vector<f_t>& max_in_row,
                      std::vector<i_t>& last_in_row,
                      std::vector<element_t<i_t, f_t>>& elements,
                      f_t& work_estimate)
{
  // Remove the pivot col
  i_t col_loop_count = 0;
  for (i_t p1 = first_in_col[pivot_j]; p1 != kNone; p1 = elements[p1].next_in_column) {
    element_t<i_t, f_t>* e = &elements[p1];
    const i_t i            = e->i;
    // Need both: last = previous-in-row (for link update when removing); last_surviving = new row
    // tail (for last_in_row[i]). They differ when the pivot is the last element in the row.
    i_t last           = kNone;
    i_t last_surviving = kNone;
#ifdef THRESHOLD_ROOK_PIVOTING
    f_t max_in_row_i = 0.0;
#endif
    i_t row_loop_count = 0;
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
      } else {
        last_surviving = p;
#ifdef THRESHOLD_ROOK_PIVOTING
        const f_t abs_entryx = std::abs(entry->x);
        if (abs_entryx > max_in_row_i) { max_in_row_i = abs_entryx; }
#endif
      }
      last = p;
      row_loop_count++;
    }
    last_in_row[i] = last_surviving;
    work_estimate += 3 * row_loop_count;
#ifdef THRESHOLD_ROOK_PIVOTING
    max_in_row[i] = max_in_row_i;
#endif
    col_loop_count++;
  }
  first_in_col[pivot_j] = kNone;
  work_estimate += 3 * col_loop_count;
}

}  // namespace
template <typename i_t, typename f_t>
i_t right_looking_lu2(const csc_matrix_t<i_t, f_t>& A,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     f_t tol,
                     const std::vector<i_t>& column_list,
                     f_t start_time,
                     std::vector<i_t>& q,
                     csc_matrix_t<i_t, f_t>& L,
                     csc_matrix_t<i_t, f_t>& U,
                     std::vector<i_t>& pinv,
                     f_t& work_estimate)
{
  raft::common::nvtx::range scope("LU::right_looking_lu");
  const i_t n = column_list.size();
  const i_t m = A.m;

  assert(A.m == n);
  assert(L.n == n);
  assert(L.m == n);
  assert(U.n == n);
  assert(U.m == n);
  assert(q.size() == n);
  assert(pinv.size() == n);


  trailing_matrix_t<i_t, f_t> trailing_matrix(A, column_list);

  csr_matrix_t<i_t, f_t> Urow(n, n, 0);  // We will store U by rows in Urow during the factorization
                                         // and translate back to U at the end
  Urow.n = Urow.m = n;
  Urow.row_start.resize(n + 1, -1);
  i_t Unz = 0;
  work_estimate += 2 * n;

  i_t Lnz = 0;
  L.x.clear();
  L.i.clear();

  std::fill(q.begin(), q.end(), -1);
  std::fill(pinv.begin(), pinv.end(), -1);
  std::vector<i_t> qinv(n);
  std::fill(qinv.begin(), qinv.end(), -1);
  work_estimate += 4 * n;

  work_estimate += trailing_matrix.record_and_clear_work_estimate_();

  i_t pivots = 0;
  for (i_t k = 0; k < n; ++k) {
    if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) {
      return CONCURRENT_HALT_RETURN;
    }
    if (toc(start_time) > settings.time_limit) { return TIME_LIMIT_RETURN; }
    // Find pivot that satisfies
    // abs(pivot) >= abstol,
    // abs(pivot) >= threshold_tol * max abs[pivot column]
    i_t pivot_i             = -1;
    i_t pivot_j             = -1;
    f_t pivot_val           = std::numeric_limits<f_t>::quiet_NaN();
    constexpr f_t pivot_tol = 1e-11;
    const f_t drop_tol      = tol == 1.0 ? 0.0 : 1e-13;
    const f_t threshold_tol = tol;

    trailing_matrix.markowitz_search(pivot_tol, threshold_tol, pivot_i, pivot_j, pivot_val);

    if (pivot_i == -1 || pivot_j == -1) { break; }
    assert(pivot_i != -1 && pivot_j != -1);

    // Pivot
    pinv[pivot_i]       = k;  // pivot_i is the kth pivot row
    q[k]                = pivot_j;
    qinv[pivot_j]       = k;
    assert(std::abs(pivot_val) >= pivot_tol);
    pivots++;

    // Cache pivot row values from column copies into dense workspace
    trailing_matrix.cache_pivot_row(pivot_i);

    // U <- [U; u^T]
    Urow.row_start[k] = Unz;
    // U(k, pivot_j) = pivot_val
    Urow.j.push_back(pivot_j);
    Urow.x.push_back(pivot_val);
    Unz++;
    // U(k, :)
    trailing_matrix.extract_row(pivot_i, pivot_j, Urow, Unz);
    work_estimate += 4 * (Unz - Urow.row_start[k]);

    // L <- [L l]
    L.col_start[k] = Lnz;
    // L(pivot_i, k) = 1
    L.i.push_back(pivot_i);
    L.x.push_back(1.0);
    Lnz++;

    // L(:, k)
    trailing_matrix.extract_column(pivot_i, pivot_j, pivot_val, L, Lnz);
    work_estimate += 4 * (Lnz - L.col_start[k]);


    trailing_matrix.update_for_pivot_removal(pivot_i, pivot_j);

    // A22 <- A22 - l u^T
    trailing_matrix.schur_complement(pivot_i, pivot_j, drop_tol, pivot_val);

    trailing_matrix.remove_pivot_row_and_column(pivot_i, pivot_j);

    trailing_matrix.garbage_collect();

    work_estimate += trailing_matrix.record_and_clear_work_estimate_();


#ifdef CHECK_MAX_IN_COLUMN
    // Check that maximum in column is maintained

#endif



  }

  trailing_matrix.print_stats();

  // Check for rank deficiency
  if (pivots < n) {
    // Complete the permutation pinv
    i_t start = pivots;
    for (i_t i = 0; i < m; ++i) {
      if (pinv[i] == -1) { pinv[i] = start++; }
    }
    work_estimate += m;

    // Finalize the permutation q. Do this by first completing the inverse permutation qinv.
    // Then invert qinv to get the final permutation q.
    start = pivots;
    for (i_t j = 0; j < n; ++j) {
      if (qinv[j] == -1) { qinv[j] = start++; }
    }
    work_estimate += n;
    inverse_permutation(qinv, q);
    work_estimate += 2 * n;

    return pivots;
  }

  // Finalize L and Urow
  L.col_start[n]    = Lnz;
  Urow.row_start[n] = Unz;

  // Fix row inidices of L for final pinv
  for (i_t p = 0; p < Lnz; ++p) {
    L.i[p] = pinv[L.i[p]];
  }
  work_estimate += 3 * Lnz;

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
  work_estimate += n;
  Urow.to_compressed_col(
    U_unpermuted);  // Convert Urow to U stored in compressed sparse column format
  work_estimate += n + Unz;
  std::vector<i_t> row_perm(n);
  work_estimate += n;
  inverse_permutation(pinv, row_perm);
  work_estimate += 2 * n;

  std::vector<i_t> identity(n);
  for (i_t k = 0; k < n; k++) {
    identity[k] = k;
  }
  work_estimate += 2 * n;

  U_unpermuted.permute_rows_and_cols(identity, q, U);
  work_estimate += 3 * U.n + 5 * Unz;

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
i_t right_looking_lu(const csc_matrix_t<i_t, f_t>& A,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     f_t tol,
                     const std::vector<i_t>& column_list,
                     f_t start_time,
                     std::vector<i_t>& q,
                     csc_matrix_t<i_t, f_t>& L,
                     csc_matrix_t<i_t, f_t>& U,
                     std::vector<i_t>& pinv,
                     f_t& work_estimate)
{
  raft::common::nvtx::range scope("LU::right_looking_lu");
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
  work_estimate += 2 * n;

  std::vector<std::vector<i_t>> col_count(
    n + 1);  // col_count[nz] is a list of columns with nz nonzeros in the active submatrix
  std::vector<std::vector<i_t>> row_count(
    n + 1);  // row_count[nz] is a list of rows with nz nonzeros in the active submatrix
  work_estimate += 2 * n;

  const i_t Bnz =
    initialize_degree_data(A, column_list, Cdegree, Rdegree, col_count, row_count, work_estimate);

  // Position arrays for O(1) degree-bucket removal (col_count and row_count each have n+1 buckets)
  std::vector<i_t> col_pos(n);  // if Cdegree[j] = nz, then j is in col_count[nz][col_pos[j]]
  std::vector<i_t> row_pos(n);  // if Rdegree[i] = nz, then i is in row_count[nz][row_pos[i]]
  initialize_bucket_positions(col_count, row_count, n, n, col_pos, row_pos);

  std::vector<element_t<i_t, f_t>> elements(Bnz);
  std::vector<i_t> first_in_row(n, kNone);
  std::vector<i_t> first_in_col(n, kNone);
  std::vector<i_t> last_in_row(n, kNone);
  work_estimate += 2 * n + Bnz;
  load_elements(
    A, column_list, Bnz, elements, first_in_row, first_in_col, last_in_row, work_estimate);

  std::vector<i_t> column_j_workspace(n, kNone);
  std::vector<i_t> row_last_workspace(n);
  std::vector<f_t> max_in_column(n);
  std::vector<f_t> max_in_row(m);
  work_estimate += 3 * n + m;

  initialize_max_in_column(first_in_col, elements, max_in_column);
  work_estimate += Bnz;

#ifdef THRESHOLD_ROOK_PIVOTING
  initialize_max_in_row(first_in_row, elements, max_in_row);
#endif

  csr_matrix_t<i_t, f_t> Urow(n, n, 0);  // We will store U by rows in Urow during the factorization
                                         // and translate back to U at the end
  Urow.n = Urow.m = n;
  Urow.row_start.resize(n + 1, -1);
  i_t Unz = 0;
  work_estimate += 2 * n;

  i_t Lnz = 0;
  L.x.clear();
  L.i.clear();

  std::fill(q.begin(), q.end(), -1);
  std::fill(pinv.begin(), pinv.end(), -1);
  std::vector<i_t> qinv(n);
  std::fill(qinv.begin(), qinv.end(), -1);
  work_estimate += 4 * n;

  i_t pivots = 0;
  for (i_t k = 0; k < n; ++k) {
    if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) {
      return CONCURRENT_HALT_RETURN;
    }
    if (toc(start_time) > settings.time_limit) { return TIME_LIMIT_RETURN; }
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
                     pivot_p,
                     work_estimate);
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
    work_estimate += 4 * (Unz - Urow.row_start[k]);

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
    work_estimate += 4 * (Lnz - L.col_start[k]);

    // Update Cdegree and col_count
    update_Cdegree_and_col_count(
      pivot_i, pivot_j, first_in_row, Cdegree, col_count, col_pos, elements, work_estimate);
    update_Rdegree_and_row_count(
      pivot_i, pivot_j, first_in_col, Rdegree, row_count, row_pos, elements, work_estimate);

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
                     last_in_row,
                     col_pos,
                     row_pos,
                     elements,
                     work_estimate);

    // Remove the pivot row
    remove_pivot_row(
      pivot_i, pivot_j, first_in_col, first_in_row, max_in_column, elements, work_estimate);
    remove_pivot_col(pivot_i,
                     pivot_j,
                     first_in_col,
                     first_in_row,
                     max_in_row,
                     last_in_row,
                     elements,
                     work_estimate);

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
    work_estimate += m;

    // Finalize the permutation q. Do this by first completing the inverse permutation qinv.
    // Then invert qinv to get the final permutation q.
    start = pivots;
    for (i_t j = 0; j < n; ++j) {
      if (qinv[j] == -1) { qinv[j] = start++; }
    }
    work_estimate += n;
    inverse_permutation(qinv, q);
    work_estimate += 2 * n;

    return pivots;
  }

  // Finalize L and Urow
  L.col_start[n]    = Lnz;
  Urow.row_start[n] = Unz;

  // Fix row inidices of L for final pinv
  for (i_t p = 0; p < Lnz; ++p) {
    L.i[p] = pinv[L.i[p]];
  }
  work_estimate += 3 * Lnz;

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
  work_estimate += n;
  Urow.to_compressed_col(
    U_unpermuted);  // Convert Urow to U stored in compressed sparse column format
  work_estimate += n + Unz;
  std::vector<i_t> row_perm(n);
  work_estimate += n;
  inverse_permutation(pinv, row_perm);
  work_estimate += 2 * n;

  std::vector<i_t> identity(n);
  for (i_t k = 0; k < n; k++) {
    identity[k] = k;
  }
  work_estimate += 2 * n;

  U_unpermuted.permute_rows_and_cols(identity, q, U);
  work_estimate += 3 * U.n + 5 * Unz;

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
  f_t work_estimate            = 0;
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

  const i_t Bnz =
    initialize_degree_data(A, column_list, Cdegree, Rdegree, col_count, row_count, work_estimate);

  // Position arrays for O(1) degree-bucket removal (col_count has m+1 buckets, row_count n+1)
  std::vector<i_t> col_pos(n);  // if Cdegree[j] = nz, then j is in col_count[nz][col_pos[j]]
  std::vector<i_t> row_pos(m);  // if Rdegree[i] = nz, then i is in row_count[nz][row_pos[i]]
  initialize_bucket_positions(col_count, row_count, m, n, col_pos, row_pos);

  std::vector<element_t<i_t, f_t>> elements(Bnz);
  std::vector<i_t> first_in_row(m, kNone);
  std::vector<i_t> first_in_col(n, kNone);
  std::vector<i_t> last_in_row(m, kNone);
  load_elements(
    A, column_list, Bnz, elements, first_in_row, first_in_col, last_in_row, work_estimate);

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
                     pivot_p,
                     work_estimate);
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
      pivot_i, pivot_j, first_in_row, Cdegree, col_count, col_pos, elements, work_estimate);
    update_Rdegree_and_row_count<i_t, f_t>(
      pivot_i, pivot_j, first_in_col, Rdegree, row_count, row_pos, elements, work_estimate);

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
                               last_in_row,
                               col_pos,
                               row_pos,
                               elements,
                               work_estimate);

    // Remove the pivot row
    remove_pivot_row<i_t, f_t>(
      pivot_i, pivot_j, first_in_col, first_in_row, max_in_column, elements, work_estimate);
    remove_pivot_col<i_t, f_t>(pivot_i,
                               pivot_j,
                               first_in_col,
                               first_in_row,
                               max_in_row,
                               last_in_row,
                               elements,
                               work_estimate);

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
    if (toc(start_time) > settings.time_limit) { return TIME_LIMIT_RETURN; }
    if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) {
      if (!settings.inside_mip) { settings.log.printf("Concurrent halt\n"); }
      return CONCURRENT_HALT_RETURN;
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
                                           double start_time,
                                           std::vector<int>& q,
                                           csc_matrix_t<int, double>& L,
                                           csc_matrix_t<int, double>& U,
                                           std::vector<int>& pinv,
                                           double& work_estimate);

template int right_looking_lu2<int, double>(const csc_matrix_t<int, double>& A,
                                            const simplex_solver_settings_t<int, double>& settings,
                                            double tol,
                                            const std::vector<int>& column_list,
                                            double start_time,
                                            std::vector<int>& q,
                                            csc_matrix_t<int, double>& L,
                                            csc_matrix_t<int, double>& U,
                                            std::vector<int>& pinv,
                                            double& work_estimate);

template int right_looking_lu_row_permutation_only<int, double>(
  const csc_matrix_t<int, double>& A,
  const simplex_solver_settings_t<int, double>& settings,
  double tol,
  double start_time,
  std::vector<int>& q,
  std::vector<int>& pinv);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
