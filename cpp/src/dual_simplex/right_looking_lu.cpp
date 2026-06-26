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

namespace cuopt::mathematical_optimization::simplex {

namespace {

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
    work_estimate_ += 4 * n;
  }

  i_t get_count(i_t k) const { return deg_[k]; }

  void update_count(i_t k, i_t new_nz)
  {
    const i_t old_nz = deg_[k];
    update_count(k, old_nz, new_nz);
  }

  const std::vector<i_t>& get_elements_with_count(i_t nz) const { return counts_[nz]; }

  // Remove k from its current bucket without re-inserting.
  // Sets deg_[k] to -1 to mark it as removed.
  void remove_from_count(i_t k)
  {
    const i_t old_nz   = deg_[k];
    const i_t p        = pos_[k];
    const i_t other    = counts_[old_nz].back();
    counts_[old_nz][p] = other;
    pos_[other]        = p;
    counts_[old_nz].pop_back();
    deg_[k] = -1;
    work_estimate_ += 6;
  }

  f_t record_and_clear_work_estimate_()
  {
    f_t tmp        = work_estimate_;
    work_estimate_ = 0;
    return tmp;
  }

 private:
  void update_count(i_t k, i_t old_nz, i_t new_nz)
  {
    const i_t p        = pos_[k];
    const i_t other    = counts_[old_nz].back();
    counts_[old_nz][p] = other;
    pos_[other]        = p;
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
  trailing_matrix_t(const csc_matrix_t<i_t, f_t>& A, const std::vector<i_t>& column_list)
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
    work_estimate_ += 4 * m_ + 2 * n_ + col_realloc_hist_.size() + row_realloc_hist_.size();

    // Allocate 2x initial size per column/row to reduce early relocations
    i_t col_nz = 2 * Bnz_;
    i_t row_nz = 2 * Bnz_;

    c_i_.resize(col_nz);
    c_x_.resize(col_nz);
    r_j_.resize(row_nz);

    i_t nz = 0;
    for (i_t i = 0; i < m_; i++) {
      row_start_[i] = nz;
      row_end_[i]   = nz;  // Temporary value used for initializing r_j_. Will be updated in loop
      i_t row_space = 2 * row_counts_.get_count(i);
      row_max_[i]   = nz + row_space;
      nz += row_space;
    }
    assert(nz == row_nz);
    work_estimate_ += 4 * m_;

    nz = 0;
    for (size_t k = 0; k < column_list.size(); k++) {
      const i_t j       = column_list[k];
      const i_t A_start = A.col_start[j];
      const i_t A_end   = A.col_start[j + 1];
      const i_t len     = A_end - A_start;
      i_t col_space     = 2 * len;
      col_max_[k]       = nz + col_space;
      col_start_[k]     = nz;
      col_end_[k]       = nz + len;
      for (i_t p = A_start; p < A_end; p++) {
        const i_t row = A.i[p];
        const f_t val = A.x[p];
        c_i_[nz]      = row;
        c_x_[nz]      = val;
        nz++;
        r_j_[row_end_[row]] = k;
        row_end_[row]++;
      }
      nz += col_space - len;  // Remaining capacity for this column
    }
    assert(nz == col_nz);
    work_estimate_ += 7 * n_ + 7 * Bnz_;

    for (i_t j = 0; j < n_; j++) {
      f_t max_in_col    = 0.0;
      const i_t c_start = col_start_[j];
      const i_t c_end   = col_end_[j];
      for (i_t p = c_start; p < c_end; p++) {
        const f_t val = std::abs(c_x_[p]);
        if (val > max_in_col) { max_in_col = val; }
      }
      max_in_column_[j] = max_in_col;
    }
    work_estimate_ += Bnz_ + 3 * n_;
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

  i_t markowitz_search(f_t pivot_tol, f_t threshold_tol, i_t& pivot_i, i_t& pivot_j, f_t& pivot_val)
  {
    f_t markowitz =
      static_cast<f_t>(m_) * static_cast<f_t>(n_);  // Upper bound on markowitz criteria
    i_t nz                 = 1;
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
        const i_t c_start    = col_start_[j];
        const i_t c_end      = col_end_[j];
        for (i_t p = c_start; p < c_end; p++) {
          const i_t i    = c_i_[p];
          const f_t val  = c_x_[p];
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
        const i_t r_end   = row_end_[i];
        for (i_t p = r_start; p < r_end; p++) {
          const i_t j = r_j_[p];
          // Look up the value from the column copy of j
          f_t val           = 0;
          const i_t c_start = col_start_[j];
          const i_t c_end   = col_end_[j];
          for (i_t q = c_start; q < c_end; q++) {
            if (c_i_[q] == i) {
              val = c_x_[q];
              break;
            }
          }
          work_estimate_ += 2 * (c_end - c_start);
          const f_t max_in_col = max_in_column_[j];
          const i_t cdeg       = col_counts_.get_count(j);
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

  void schur_complement(i_t pivot_i, i_t pivot_j, f_t drop_tol, f_t pivot_val)
  {
    // Step 1: Cache the pivot column into dense workspaces.
    // pivot_col_val_[i] = l_i = a(i, pivot_j) / pivot_val  for each row i != pivot_i
    // pivot_col_mark_[i] = 1 if row i is in the pivot column
    // pivot_col_index_ = list of i such that pivot_col_mark[i] == 1
    i_t pivot_col_count     = 0;
    const i_t c_pivot_start = col_start_[pivot_j];
    const i_t c_pivot_end   = col_end_[pivot_j];
    for (i_t p = c_pivot_start; p < c_pivot_end; p++) {
      const i_t i = c_i_[p];
      if (i == pivot_i) { continue; }
      const f_t li       = c_x_[p] / pivot_val;
      pivot_col_val_[i]  = li;
      pivot_col_mark_[i] = 1;
      pivot_col_index_.push_back(i);
      pivot_col_count++;
    }
    work_estimate_ += 5 * (c_pivot_end - c_pivot_start);

    // Step 2: For each column j in the pivot row, update existing entries and insert fill.
    const i_t r_pivot_start = row_start_[pivot_i];
    const i_t r_pivot_end   = row_end_[pivot_i];
    for (i_t p0 = r_pivot_start; p0 < r_pivot_end; p0++) {
      const i_t j = r_j_[p0];
      if (j == pivot_j) { continue; }
      const f_t uj = pivot_row_val_[j];

      // Step 2a: Scan column j, update existing entries, and count fill-in.
      // For each entry (i, j) that also appears in the pivot column, update it.
      // Simultaneously, unmark pivot_col_mark_[i] for matched entries, so that
      // after the scan, the still-marked entries are the fill-ins.
      i_t n_fillin      = pivot_col_count;
      i_t n_cancel      = 0;
      const i_t c_start = col_start_[j];
      const i_t c_end   = col_end_[j];
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
      work_estimate_ += 2 * (c_end - c_start) + 6 * (pivot_col_count - n_fillin);

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
            work_estimate_ += 2 * (row_end_[dead_row] - r_start) + 4;
            // Update row degree
            const i_t rdeg = row_counts_.get_count(dead_row);
            row_counts_.update_count(dead_row, rdeg - 1);
          }
        }
        work_estimate_ += 2 * (new_end - col_start_[j]);
        i_t old_count = col_end_[j] - col_start_[j];
        col_end_[j]   = new_end;
        i_t new_count = new_end - col_start_[j] - 1;  // -1 for pivot row removal
        // Update column degree for cancellations
        if (new_count != old_count) { col_counts_.update_count(j, new_count); }
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
            const f_t val     = pivot_col_val_[i] * uj;
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
        i_t new_cdeg = col_end_[j] - col_start_[j] - 1;  // -1 for pivot row removal
        if (new_cdeg != col_counts_.get_count(j)) { col_counts_.update_count(j, new_cdeg); }
      }

      // Step 2e: Reset all pivot column marks back to 1 for the next column.
      // Some marks were cleared to 0 during the scan of column j (matched entries).
      // We restore them by iterating the pivot column index list. So that we are
      // prepared to process the next column.
      for (i_t k = 0; k < pivot_col_count; k++) {
        pivot_col_mark_[pivot_col_index_[k]] = 1;
      }
      work_estimate_ += 2 * pivot_col_count;
    }

    // Step 3: Clear the pivot column workspaces.
    for (i_t k = 0; k < pivot_col_count; k++) {
      const i_t i        = pivot_col_index_[k];
      pivot_col_val_[i]  = 0;
      pivot_col_mark_[i] = 0;
    }
    work_estimate_ += 2 * pivot_col_count;
    pivot_col_index_.clear();
  }

  // Populate the dense pivot_row_val_ workspace by scanning column representation
  // for each column j that appears in the pivot row.
  // Must be called before extract_row() and schur_complement().
  // Cleared by remove_pivot_row_and_column().
  void cache_pivot_row(i_t pivot_i)
  {
    const i_t r_start = row_start_[pivot_i];
    const i_t r_end   = row_end_[pivot_i];
    for (i_t p = r_start; p < r_end; p++) {
      const i_t j       = r_j_[p];
      const i_t c_start = col_start_[j];
      const i_t c_end   = col_end_[j];
      i_t q;
      for (q = c_start; q < c_end; q++) {
        if (c_i_[q] == pivot_i) {
          pivot_row_val_[j] = c_x_[q];
          break;
        }
      }
      work_estimate_ += 2 * (q - c_start);
    }
    work_estimate_ += 3 * (r_end - r_start) + 2;
  }

  void remove_pivot_row_and_column(i_t pivot_i, i_t pivot_j)
  {
    // Iterate over the pivot row
    const i_t r_pivot_start = row_start_[pivot_i];
    const i_t r_pivot_end   = row_end_[pivot_i];
    for (i_t p = r_pivot_start; p < r_pivot_end; p++) {
      const i_t j = r_j_[p];
      // Clear the cached pivot row value for this column
      pivot_row_val_[j] = 0;
      // Skip the pivot column: its storage and count will be set to 0 at the end
      if (j == pivot_j) { continue; }
      // Remove pivot_i from each column j in the pivot row
      f_t max_in_col = 0.0;

      const i_t prev_col_end = col_end_[j];
      for (i_t q = col_start_[j]; q < col_end_[j]; q++) {
        const i_t i = c_i_[q];
        if (i == pivot_i) {
          // Swap with the last element in the column
          i_t other_i = c_i_[col_end_[j] - 1];
          f_t other_x = c_x_[col_end_[j] - 1];
          c_i_[q]     = other_i;
          c_x_[q]     = other_x;
          // Update col_end_[j]
          col_end_[j]--;
          q--;
          continue;
        } else {
          const f_t val = std::abs(c_x_[q]);
          if (val > max_in_col) { max_in_col = val; }
        }
      }
      work_estimate_ += 3 * (prev_col_end - col_start_[j]) + 7;
      max_in_column_[j] = max_in_col;
    }
    work_estimate_ += 4 * (r_pivot_end - r_pivot_start);

    // Iterate over the pivot column: decrement row degrees and remove pivot_j
    // from each row. Skip the pivot row itself we remove it at the end.
    const i_t c_start = col_start_[pivot_j];
    const i_t c_end   = col_end_[pivot_j];
    for (i_t p = c_start; p < c_end; p++) {
      const i_t i = c_i_[p];
      if (i == pivot_i) { continue; }

      const i_t rdeg = row_counts_.get_count(i);
      row_counts_.update_count(i, rdeg - 1);

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
      work_estimate_ += 2 * (q - row_start_[i]) + 4;
    }
    work_estimate_ += 4 * (c_end - c_start);

    // Mark pivot column and pivot row as empty so garbage collection skips them
    col_end_[pivot_j] = col_start_[pivot_j];
    col_counts_.update_count(pivot_j, 0);
    row_end_[pivot_i] = row_start_[pivot_i];
    row_counts_.update_count(pivot_i, 0);
  }

  void extract_row(i_t pivot_i, i_t pivot_j, csr_matrix_t<i_t, f_t>& Urow, i_t& Unz)
  {
    // U(k, :)
    const i_t r_pivot_start = row_start_[pivot_i];
    const i_t r_pivot_end   = row_end_[pivot_i];
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
    const i_t c_pivot_end   = col_end_[pivot_j];
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
      std::vector<i_t> new_c_i;
      std::vector<f_t> new_c_x;
      new_c_i.reserve(c_i_.size() - unused_col_nz_);
      new_c_x.reserve(c_x_.size() - unused_col_nz_);
      for (i_t j = 0; j < n_; j++) {
        const i_t new_start = static_cast<i_t>(new_c_i.size());
        const i_t c_start   = col_start_[j];
        const i_t c_end     = col_end_[j];
        const i_t col_size  = c_end - c_start;
        for (i_t p = c_start; p < c_end; p++) {
          new_c_i.push_back(c_i_[p]);
          new_c_x.push_back(c_x_[p]);
        }
        col_start_[j] = new_start;
        col_end_[j]   = static_cast<i_t>(new_c_i.size());
        // Reserve space equal to current size (doubling strategy)
        for (i_t s = 0; s < col_size; s++) {
          new_c_i.push_back(kNone);
          new_c_x.push_back(0.0);
        }
        work_estimate_ += 4 * col_size;
        col_max_[j] = static_cast<i_t>(new_c_i.size());
      }
      work_estimate_ += 6 * n_;
      c_i_ = std::move(new_c_i);
      c_x_ = std::move(new_c_x);

      unused_col_nz_ = 0;
    }

    if (unused_row_nz_ > max_unused_fraction * static_cast<f_t>(r_j_.size())) {
      std::vector<i_t> new_r_j;
      new_r_j.reserve(r_j_.size() - unused_row_nz_);
      for (i_t i = 0; i < m_; i++) {
        const i_t new_start = static_cast<i_t>(new_r_j.size());
        const i_t r_start   = row_start_[i];
        const i_t r_end     = row_end_[i];
        const i_t row_size  = r_end - r_start;
        for (i_t p = r_start; p < r_end; p++) {
          new_r_j.push_back(r_j_[p]);
        }
        row_start_[i] = new_start;
        row_end_[i]   = static_cast<i_t>(new_r_j.size());
        // Reserve space equal to current size (doubling strategy)
        for (i_t s = 0; s < row_size; s++) {
          new_r_j.push_back(kNone);
        }
        row_max_[i] = static_cast<i_t>(new_r_j.size());
        work_estimate_ += 2 * row_size;
      }
      work_estimate_ += 6 * m_;
      r_j_           = std::move(new_r_j);
      unused_row_nz_ = 0;
    }
  }

  void print_stats()
  {
#ifdef PRINT_STATS
    printf("Column hits: %.1f%%, Column misses: %.1f%%, Row hits: %.1f%%, Row misses: %.1f%%\n",
           100.0 * static_cast<f_t>(col_hits_) / static_cast<f_t>(col_hits_ + col_miss_),
           100.0 * static_cast<f_t>(col_miss_) / static_cast<f_t>(col_hits_ + col_miss_),
           100.0 * static_cast<f_t>(row_hits_) / static_cast<f_t>(row_hits_ + row_miss_),
           100.0 * static_cast<f_t>(row_miss_) / static_cast<f_t>(row_hits_ + row_miss_));

    printf("Column reallocation histogram (shortfall -> count):\n");
    for (size_t k = 0; k < col_realloc_hist_.size(); k++) {
      if (col_realloc_hist_[k] > 0) { printf("  %4zu: %d\n", k, col_realloc_hist_[k]); }
    }

    printf("Row reallocation histogram (shortfall -> count):\n");
    for (size_t k = 0; k < row_realloc_hist_.size(); k++) {
      if (row_realloc_hist_[k] > 0) { printf("  %4zu: %d\n", k, row_realloc_hist_[k]); }
    }

    f_t ci_mb = static_cast<f_t>(c_i_.size() * sizeof(i_t)) / (1024.0 * 1024.0);
    f_t cx_mb = static_cast<f_t>(c_x_.size() * sizeof(f_t)) / (1024.0 * 1024.0);
    f_t rj_mb = static_cast<f_t>(r_j_.size() * sizeof(i_t)) / (1024.0 * 1024.0);
    printf("Memory: c_i_ = %.2f MB, c_x_ = %.2f MB, r_j_ = %.2f MB, total = %.2f MB\n",
           ci_mb,
           cx_mb,
           rj_mb,
           ci_mb + cx_mb + rj_mb);
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
    const i_t c_end   = col_end_[j];
    i_t current_size  = c_end - c_start;
    unused_col_nz_ += current_size;
    i_t new_start = c_i_.size();
    for (i_t p = c_start; p < c_end; p++) {
      c_i_.push_back(c_i_[p]);
      c_x_.push_back(c_x_[p]);
    }
    work_estimate_ += 2 * (c_end - c_start);
    col_start_[j] = new_start;
    col_end_[j]   = c_i_.size();
    // Reserve space using doubling strategy to reduce future relocations
    i_t extra = std::max(current_size, needed);
    for (i_t k = 0; k < extra; k++) {
      c_i_.push_back(kNone);
      c_x_.push_back(0.0);
    }
    work_estimate_ += 2 * extra;
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
    const i_t r_end   = row_end_[i];
    i_t current_size  = r_end - r_start;
    unused_row_nz_ += current_size;
    i_t new_start = r_j_.size();
    for (i_t p = r_start; p < r_end; p++) {
      r_j_.push_back(r_j_[p]);
    }
    work_estimate_ += (r_end - r_start);
    row_start_[i] = new_start;
    row_end_[i]   = r_j_.size();
    // Reserve space using doubling strategy to reduce future relocations
    i_t extra = std::max(current_size, needed);
    for (i_t k = 0; k < extra; k++) {
      r_j_.push_back(kNone);
    }
    work_estimate_ += extra;
    row_max_[i] = r_j_.size();
    work_estimate_ += 9;
  }

  std::vector<i_t> compute_column_degree(const csc_matrix_t<i_t, f_t>& A,
                                         const std::vector<i_t>& column_list)
  {
    const i_t n = column_list.size();
    std::vector<i_t> Cdegree(n);
    for (i_t k = 0; k < n; k++) {
      const i_t j       = column_list[k];
      const i_t A_start = A.col_start[j];
      const i_t A_end   = A.col_start[j + 1];
      Cdegree[k]        = A_end - A_start;
    }
    work_estimate_ += 4 * n;
    return Cdegree;
  }

  std::vector<i_t> compute_row_degree(const csc_matrix_t<i_t, f_t>& A,
                                      const std::vector<i_t>& column_list,
                                      i_t& Bnz)
  {
    std::vector<i_t> Rdegree(A.m, 0);
    Bnz         = 0;
    const i_t n = column_list.size();
    for (i_t k = 0; k < n; k++) {
      const i_t j         = column_list[k];
      const i_t col_start = A.col_start[j];
      const i_t col_end   = A.col_start[j + 1];
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

  std::vector<i_t> c_i_;  // row indices (indexed by col_start_[j] to col_end_[j])
  std::vector<f_t> c_x_;  // coefficients (indexed by col_start_[j] to col_end_[j])

  // The representation of the matrix by row (index only, no values)
  std::vector<i_t> row_start_;
  std::vector<i_t> row_end_;
  std::vector<i_t> row_max_;

  std::vector<i_t> r_j_;  // column indices (indexed by row_start_[i] to row_end_[i])

  std::vector<f_t>
    max_in_column_;  // max_in_column_[j] is absolute value of the maximum coefficient in column j

  std::vector<f_t> pivot_row_val_;  // dense workspace of size n_; caches pivot row values

  std::vector<f_t>
    pivot_col_val_;  // dense workspace of size m_; caches L multipliers for pivot column
  std::vector<char>
    pivot_col_mark_;  // dense workspace of size m_; 1 if row i is in the pivot column
  std::vector<i_t>
    pivot_col_index_;  // sparse list of row indices in the pivot column (excl. pivot_i)

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

  std::vector<i_t>
    col_realloc_hist_;  // col_realloc_hist_[k] = number of column relocations with shortfall k
  std::vector<i_t>
    row_realloc_hist_;  // row_realloc_hist_[k] = number of row relocations with shortfall k
};

}  // namespace
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
    pinv[pivot_i] = k;  // pivot_i is the kth pivot row
    q[k]          = pivot_j;
    qinv[pivot_j] = k;
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

    // A22 <- A22 - l u^T
    trailing_matrix.schur_complement(pivot_i, pivot_j, drop_tol, pivot_val);

    trailing_matrix.remove_pivot_row_and_column(pivot_i, pivot_j);

    trailing_matrix.garbage_collect();

    work_estimate += trailing_matrix.record_and_clear_work_estimate_();
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
  assert(q.size() == n);
  (void)tol;  // Unused; kept for API compatibility with right_looking_lu_row_permutation_only

  std::vector<i_t> column_list(n);
  for (i_t k = 0; k < n; ++k) {
    column_list[k] = k;
  }

  trailing_matrix_t<i_t, f_t> trailing_matrix(A, column_list);

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
    f_t pivot_val               = std::numeric_limits<f_t>::quiet_NaN();
    constexpr f_t pivot_tol     = 1e-9;
    constexpr f_t drop_tol      = 1e-14;
    constexpr f_t threshold_tol = 1.0 / 10.0;
    trailing_matrix.markowitz_search(pivot_tol, threshold_tol, pivot_i, pivot_j, pivot_val);
    if (pivot_i == -1 || pivot_j == -1) {
      settings.log.debug("Breaking can't find a pivot %d\n", k);
      break;
    }
    // Pivot
    pinv[pivot_i] = k;  // pivot_i is the kth pivot row
    q[k]          = pivot_j;
    qinv[pivot_j] = k;
    assert(std::abs(pivot_val) >= pivot_tol);
    pivots++;

    trailing_matrix.cache_pivot_row(pivot_i);

    trailing_matrix.schur_complement(pivot_i, pivot_j, drop_tol, pivot_val);

    trailing_matrix.remove_pivot_row_and_column(pivot_i, pivot_j);

    trailing_matrix.garbage_collect();

    if (toc(last_print) > 10.0) {
      settings.log.printf(
        "Right-looking LU factorization: Pivots %d m %d n %d in "
        "%.2f seconds\n",
        pivots,
        m,
        n,
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

// =============================================================================
// Symmetric positive semidefinite LDL^T factorization with Markowitz pivoting.
// Computes P * A * P^T = L * D * L^T where:
//   - A is symmetric PSD (lower triangle stored in CSC)
//   - P is a symmetric fill-reducing permutation (from Markowitz)
//   - L is unit lower triangular (stored in CSC)
//   - D is diagonal (stored as a vector)
// =============================================================================

namespace {

// Represents the lower triangle (including diagonal) of a symmetric trailing matrix
// during right-looking LDL^T factorization.
// Stores column representation of the lower triangle and row representation (index only)
// of the lower triangle. Since the matrix is symmetric, "row i" stores columns j <= i
// that have a nonzero in position (i, j).
template <typename i_t, typename f_t>
class symmetric_trailing_matrix_t {
 public:
  // Construct from a symmetric matrix stored as lower triangle in CSC format.
  // A.col_start[j]..A.col_start[j+1]-1 contain entries (i, j) with i >= j.
  symmetric_trailing_matrix_t(const csc_matrix_t<i_t, f_t>& A)
    : n_(A.n),
      Bnz_(0),
      work_estimate_(0),
      col_start_(n_),
      col_end_(n_),
      col_max_(n_),
      row_start_(n_),
      row_end_(n_),
      row_max_(n_),
      diag_(n_, 0.0),
      pivot_col_val_(n_, 0.0),
      pivot_col_mark_(n_, 0),
      counts_(compute_degree(A), n_),
      unused_col_nz_(0),
      unused_row_nz_(0)
  {
    // Count total nonzeros (lower triangle including diagonal)
    for (i_t j = 0; j < n_; j++) {
      Bnz_ += A.col_start[j + 1] - A.col_start[j];
    }
    work_estimate_ += 2 * n_;

    // Allocate 2x initial size for column and row storage
    i_t col_nz = 2 * Bnz_;
    i_t row_nz = 2 * Bnz_;

    c_i_.resize(col_nz);
    c_x_.resize(col_nz);
    r_j_.resize(row_nz);

    // Initialize row storage pointers
    // Row i stores columns j < i (off-diagonal entries in row i of lower triangle).
    // The diagonal is stored separately in diag_.
    // Row degree = number of off-diagonal entries in row i of the lower triangle.
    i_t nz = 0;
    for (i_t i = 0; i < n_; i++) {
      row_start_[i] = nz;
      row_end_[i]   = nz;
      // The degree from compute_degree counts off-diag entries in column i (below diagonal)
      // plus off-diag entries in row i (same thing by symmetry).
      // For row storage, we store columns j < i that appear in column j at row i.
      // We'll compute this during column init below. For now, reserve based on degree.
      i_t row_space = 2 * counts_.get_count(i);
      row_max_[i]   = nz + row_space;
      nz += row_space;
    }
    // Resize row storage if needed
    if (nz > row_nz) { r_j_.resize(nz); }
    work_estimate_ += 4 * n_;

    // Initialize column storage and populate row indices
    nz = 0;
    for (i_t j = 0; j < n_; j++) {
      const i_t A_start = A.col_start[j];
      const i_t A_end   = A.col_start[j + 1];
      // Count off-diagonal entries (entries below diagonal)
      i_t off_diag_count = 0;
      for (i_t p = A_start; p < A_end; p++) {
        if (A.i[p] == j) {
          diag_[j] = A.x[p];
        } else {
          off_diag_count++;
        }
      }
      i_t col_space = 2 * std::max(off_diag_count, i_t(1));
      col_max_[j]   = nz + col_space;
      col_start_[j] = nz;
      col_end_[j]   = nz;

      // Store only off-diagonal entries in the column (i > j)
      for (i_t p = A_start; p < A_end; p++) {
        const i_t row = A.i[p];
        const f_t val = A.x[p];
        if (row == j) { continue; }  // diagonal handled above
        assert(row > j);             // lower triangle: row > col for off-diag
        c_i_[col_end_[j]] = row;
        c_x_[col_end_[j]] = val;
        col_end_[j]++;

        // Also add to row storage: row `row` has an entry in column j
        ensure_row_space(row, 1);
        r_j_[row_end_[row]] = j;
        row_end_[row]++;
      }
      nz += col_space;
    }
    // Resize column storage to actual allocated size
    if (nz > col_nz) {
      c_i_.resize(nz);
      c_x_.resize(nz);
    }
    work_estimate_ += 7 * n_ + 7 * Bnz_;
  }

  f_t record_and_clear_work_estimate_()
  {
    const f_t counts_work = counts_.record_and_clear_work_estimate_();
    work_estimate_ += counts_work;
    f_t tmp        = work_estimate_;
    work_estimate_ = 0;
    return tmp;
  }

  // Symmetric Markowitz search: find pivot p on diagonal with |diag_[p]| >= pivot_tol
  // minimizing degree. Within a fixed degree, select the largest |diag[p]|.
  // Degree here is the number of off-diagonal neighbors.
  // Returns the number of candidates searched.
  i_t symmetric_markowitz_search(f_t pivot_tol, i_t& pivot_p, f_t& pivot_val)
  {
    i_t best_degree = n_;
    i_t nsearch     = 0;

    for (i_t nz = 0; nz <= best_degree; nz++) {
      const auto& elements = counts_.get_elements_with_count(nz);
      for (const i_t p : elements) {
        const f_t d = diag_[p];
        if (std::abs(d) >= pivot_tol) {
          if (nz < best_degree || (nz == best_degree && std::abs(d) > std::abs(pivot_val))) {
            best_degree = nz;
            pivot_p     = p;
            pivot_val   = d;
          }
        }
        nsearch++;
      }
      work_estimate_ += 4 * static_cast<i_t>(elements.size());
      // Once we've found a pivot at degree nz, no need to search higher degrees
      if (pivot_p != -1 && nz == best_degree) { break; }
    }
    return nsearch;
  }

  // Symmetric Schur complement: update A <- A - d * l * l^T (lower triangle only)
  // where l_i = A(i, pivot_p) / d for all i != pivot_p, d = diag_[pivot_p].
  // The full pivot vector l comes from:
  //   - Column pivot_p entries (rows i > pivot_p): stored directly
  //   - Row pivot_p entries (cols j < pivot_p): these represent A(pivot_p, j) = A(j, pivot_p)
  //     by symmetry, but are stored in column j at row pivot_p.
  void symmetric_schur_complement(i_t pivot_p, f_t drop_tol, f_t pivot_val)
  {
    // Step 1: Build the full pivot vector from both column (below) and row (left).
    // For i > pivot_p: l_i = A(i, pivot_p) / pivot_val, found in column pivot_p
    // For j < pivot_p: l_j = A(pivot_p, j) / pivot_val = A(j, pivot_p) / pivot_val
    //   A(pivot_p, j) is stored in column j at row pivot_p (lower triangle: pivot_p > j)
    i_t pivot_col_count = 0;

    // From column storage: entries (i, pivot_p) with i > pivot_p
    const i_t c_pivot_start = col_start_[pivot_p];
    const i_t c_pivot_end   = col_end_[pivot_p];
    for (i_t p = c_pivot_start; p < c_pivot_end; p++) {
      const i_t i        = c_i_[p];
      const f_t li       = c_x_[p] / pivot_val;
      pivot_col_val_[i]  = li;
      pivot_col_mark_[i] = 1;
      pivot_col_index_.push_back(i);
      pivot_col_count++;
    }

    // From row storage: entries (pivot_p, j) with j < pivot_p
    // The value A(pivot_p, j) is stored in column j at row pivot_p.
    const i_t r_pivot_start = row_start_[pivot_p];
    const i_t r_pivot_end   = row_end_[pivot_p];
    for (i_t rp = r_pivot_start; rp < r_pivot_end; rp++) {
      const i_t j = r_j_[rp];
      // Look up A(pivot_p, j) from column j
      f_t val = 0;
      for (i_t q = col_start_[j]; q < col_end_[j]; q++) {
        if (c_i_[q] == pivot_p) {
          val = c_x_[q];
          break;
        }
      }
      const f_t lj       = val / pivot_val;
      pivot_col_val_[j]  = lj;
      pivot_col_mark_[j] = 1;
      pivot_col_index_.push_back(j);
      pivot_col_count++;
    }
    work_estimate_ += 5 * (c_pivot_end - c_pivot_start) + 5 * (r_pivot_end - r_pivot_start);

    // Step 2: For each node j in the pivot vector, update the trailing matrix.
    // The update is: A(i, j) -= pivot_val * l_i * l_j for all pairs (i, j) with i >= j
    // that are both in the pivot vector. Also update diagonals.
    for (i_t k = 0; k < pivot_col_count; k++) {
      const i_t j  = pivot_col_index_[k];
      const f_t lj = pivot_col_val_[j];

      // Update diagonal: A(j,j) -= pivot_val * lj * lj
      diag_[j] -= pivot_val * lj * lj;

      // Update off-diagonal entries in column j: A(i, j) -= pivot_val * l_i * l_j
      // for i > j where l_i != 0 (i.e., i is in pivot vector)
      // Column j stores entries with row > j in lower triangle.

      // Count how many pivot vector entries with index > j exist (potential updates/fills)
      i_t n_pivot_entries_below_j = 0;
      for (i_t m = 0; m < pivot_col_count; m++) {
        if (pivot_col_index_[m] > j) { n_pivot_entries_below_j++; }
      }

      // Scan existing entries in column j and update those that are in pivot vector
      const i_t c_start = col_start_[j];
      const i_t c_end   = col_end_[j];
      i_t n_updated     = 0;
      i_t n_cancel      = 0;
      for (i_t q = c_start; q < c_end; q++) {
        const i_t i = c_i_[q];
        if (pivot_col_mark_[i]) {
          // Entry (i, j) exists and i is in pivot vector: update
          const f_t li = pivot_col_val_[i];
          c_x_[q] -= pivot_val * li * lj;
          if (std::abs(c_x_[q]) < drop_tol) {
            c_x_[q] = 0;
            n_cancel++;
          }
          n_updated++;
        }
      }
      work_estimate_ += 4 * (c_end - c_start);
      i_t n_fillin = n_pivot_entries_below_j - n_updated;

      // Step 2b: Remove cancellations
      if (n_cancel > 0) {
        i_t new_end = col_start_[j];
        for (i_t q = col_start_[j]; q < col_end_[j]; q++) {
          if (c_x_[q] != 0) {
            c_i_[new_end] = c_i_[q];
            c_x_[new_end] = c_x_[q];
            new_end++;
          } else {
            const i_t dead_row = c_i_[q];
            // Remove column j from row dead_row
            for (i_t rp2 = row_start_[dead_row]; rp2 < row_end_[dead_row]; rp2++) {
              if (r_j_[rp2] == j) {
                r_j_[rp2] = r_j_[row_end_[dead_row] - 1];
                row_end_[dead_row]--;
                break;
              }
            }
            // Update degree for dead_row
            const i_t rdeg = counts_.get_count(dead_row);
            if (rdeg > 0) { counts_.update_count(dead_row, rdeg - 1); }
            work_estimate_ += 6;
          }
        }
        col_end_[j] = new_end;
      }

      // Step 2c: Insert fill-in entries
      if (n_fillin > 0) {
        ensure_col_space(j, n_fillin);
        for (i_t m = 0; m < pivot_col_count; m++) {
          const i_t i = pivot_col_index_[m];
          if (i <= j) { continue; }  // only lower triangle: i > j
          // Check if (i, j) already exists
          bool found = false;
          for (i_t q = col_start_[j]; q < col_end_[j]; q++) {
            if (c_i_[q] == i) {
              found = true;
              break;
            }
          }
          if (found) { continue; }

          // Insert fill-in: A(i, j) = -pivot_val * l_i * l_j
          const f_t li  = pivot_col_val_[i];
          const f_t val = -pivot_val * li * lj;
          if (std::abs(val) < drop_tol) { continue; }

          c_i_[col_end_[j]] = i;
          c_x_[col_end_[j]] = val;
          col_end_[j]++;

          // Insert into row storage: row i gets column j
          ensure_row_space(i, 1);
          r_j_[row_end_[i]] = j;
          row_end_[i]++;

          // Update degree for row i
          const i_t rdeg = counts_.get_count(i);
          counts_.update_count(i, rdeg + 1);
          work_estimate_ += 10;
        }
      }

      // Update degree for node j
      i_t col_entries = col_end_[j] - col_start_[j];
      i_t row_entries = row_end_[j] - row_start_[j];
      i_t total_deg   = col_entries + row_entries;
      if (total_deg != counts_.get_count(j)) { counts_.update_count(j, total_deg); }
    }

    // Step 3: Clear pivot vector workspaces
    for (i_t k = 0; k < pivot_col_count; k++) {
      const i_t i        = pivot_col_index_[k];
      pivot_col_val_[i]  = 0;
      pivot_col_mark_[i] = 0;
    }
    pivot_col_index_.clear();
    work_estimate_ += 2 * pivot_col_count;
  }

  // Extract the full pivot vector as the L column (divided by pivot value).
  // The L column for pivot p includes ALL off-diagonal entries of the full symmetric
  // column p: entries below (from column storage) AND entries above (from row storage,
  // looked up in other columns). Returns entries in original row indices.
  void extract_column(i_t pivot_p, f_t pivot_val, csc_matrix_t<i_t, f_t>& L, i_t& Lnz)
  {
    // Entries below pivot: stored directly in column pivot_p
    const i_t c_start = col_start_[pivot_p];
    const i_t c_end   = col_end_[pivot_p];
    for (i_t p = c_start; p < c_end; p++) {
      const i_t i     = c_i_[p];
      const f_t l_val = c_x_[p] / pivot_val;
      L.i.push_back(i);
      L.x.push_back(l_val);
      Lnz++;
    }

    // Entries above pivot: row pivot_p stores columns j < pivot_p.
    // The value A(pivot_p, j) is stored in column j at row pivot_p.
    const i_t r_start = row_start_[pivot_p];
    const i_t r_end   = row_end_[pivot_p];
    for (i_t rp = r_start; rp < r_end; rp++) {
      const i_t j = r_j_[rp];
      // Look up A(pivot_p, j) from column j
      f_t val = 0;
      for (i_t q = col_start_[j]; q < col_end_[j]; q++) {
        if (c_i_[q] == pivot_p) {
          val = c_x_[q];
          break;
        }
      }
      const f_t l_val = val / pivot_val;
      L.i.push_back(j);
      L.x.push_back(l_val);
      Lnz++;
    }
    work_estimate_ += 4 * (c_end - c_start) + 5 * (r_end - r_start);
  }

  // Remove the pivot node from the trailing matrix: remove column pivot_p and row pivot_p.
  void remove_pivot(i_t pivot_p)
  {
    // Remove pivot_p from all rows that reference it via column storage
    // Column pivot_p has entries at rows i > pivot_p. For each such row i,
    // remove pivot_p from the row-index list of row i.
    const i_t c_start = col_start_[pivot_p];
    const i_t c_end   = col_end_[pivot_p];
    for (i_t p = c_start; p < c_end; p++) {
      const i_t i = c_i_[p];
      // Remove column pivot_p from row i
      for (i_t rp = row_start_[i]; rp < row_end_[i]; rp++) {
        if (r_j_[rp] == pivot_p) {
          r_j_[rp] = r_j_[row_end_[i] - 1];
          row_end_[i]--;
          break;
        }
      }
      // Update degree for row i: decrement by 1 (lost column pivot_p)
      const i_t deg = counts_.get_count(i);
      if (deg > 0) { counts_.update_count(i, deg - 1); }
    }
    work_estimate_ += 6 * (c_end - c_start);

    // Remove pivot_p from all columns that row pivot_p references.
    // Row pivot_p stores columns j < pivot_p that have an entry at row pivot_p.
    const i_t r_start = row_start_[pivot_p];
    const i_t r_end   = row_end_[pivot_p];
    for (i_t rp = r_start; rp < r_end; rp++) {
      const i_t j = r_j_[rp];
      // Remove row pivot_p from column j
      for (i_t q = col_start_[j]; q < col_end_[j]; q++) {
        if (c_i_[q] == pivot_p) {
          c_i_[q] = c_i_[col_end_[j] - 1];
          c_x_[q] = c_x_[col_end_[j] - 1];
          col_end_[j]--;
          break;
        }
      }
      // Update degree for column j: lost row pivot_p
      i_t col_entries = col_end_[j] - col_start_[j];
      i_t row_entries = row_end_[j] - row_start_[j];
      i_t total_deg   = col_entries + row_entries;
      if (total_deg != counts_.get_count(j)) { counts_.update_count(j, total_deg); }
    }
    work_estimate_ += 6 * (r_end - r_start);

    // Mark pivot as eliminated
    col_end_[pivot_p] = col_start_[pivot_p];
    row_end_[pivot_p] = row_start_[pivot_p];
    counts_.remove_from_count(pivot_p);
  }

  void garbage_collect(f_t max_unused_fraction = 0.90)
  {
    if (unused_col_nz_ > max_unused_fraction * static_cast<f_t>(c_i_.size())) {
      std::vector<i_t> new_c_i;
      std::vector<f_t> new_c_x;
      new_c_i.reserve(c_i_.size() - unused_col_nz_);
      new_c_x.reserve(c_x_.size() - unused_col_nz_);
      for (i_t j = 0; j < n_; j++) {
        const i_t new_start = static_cast<i_t>(new_c_i.size());
        const i_t c_start   = col_start_[j];
        const i_t c_end     = col_end_[j];
        const i_t col_size  = c_end - c_start;
        for (i_t p = c_start; p < c_end; p++) {
          new_c_i.push_back(c_i_[p]);
          new_c_x.push_back(c_x_[p]);
        }
        col_start_[j] = new_start;
        col_end_[j]   = static_cast<i_t>(new_c_i.size());
        for (i_t s = 0; s < col_size; s++) {
          new_c_i.push_back(kNone);
          new_c_x.push_back(0.0);
        }
        col_max_[j] = static_cast<i_t>(new_c_i.size());
        work_estimate_ += 4 * col_size;
      }
      work_estimate_ += 6 * n_;
      c_i_           = std::move(new_c_i);
      c_x_           = std::move(new_c_x);
      unused_col_nz_ = 0;
    }

    if (unused_row_nz_ > max_unused_fraction * static_cast<f_t>(r_j_.size())) {
      std::vector<i_t> new_r_j;
      new_r_j.reserve(r_j_.size() - unused_row_nz_);
      for (i_t i = 0; i < n_; i++) {
        const i_t new_start = static_cast<i_t>(new_r_j.size());
        const i_t r_start   = row_start_[i];
        const i_t r_end     = row_end_[i];
        const i_t row_size  = r_end - r_start;
        for (i_t p = r_start; p < r_end; p++) {
          new_r_j.push_back(r_j_[p]);
        }
        row_start_[i] = new_start;
        row_end_[i]   = static_cast<i_t>(new_r_j.size());
        for (i_t s = 0; s < row_size; s++) {
          new_r_j.push_back(kNone);
        }
        row_max_[i] = static_cast<i_t>(new_r_j.size());
        work_estimate_ += 2 * row_size;
      }
      work_estimate_ += 6 * n_;
      r_j_           = std::move(new_r_j);
      unused_row_nz_ = 0;
    }
  }

 private:
  bool ensure_col_space(i_t j, i_t needed)
  {
    if (col_end_[j] + needed <= col_max_[j]) { return false; }
    const i_t c_start = col_start_[j];
    const i_t c_end   = col_end_[j];
    i_t current_size  = c_end - c_start;
    unused_col_nz_ += current_size;
    i_t new_start = c_i_.size();
    for (i_t p = c_start; p < c_end; p++) {
      c_i_.push_back(c_i_[p]);
      c_x_.push_back(c_x_[p]);
    }
    work_estimate_ += 2 * (c_end - c_start);
    col_start_[j] = new_start;
    col_end_[j]   = c_i_.size();
    i_t extra     = std::max(current_size, needed);
    for (i_t k = 0; k < extra; k++) {
      c_i_.push_back(kNone);
      c_x_.push_back(0.0);
    }
    work_estimate_ += 2 * extra;
    col_max_[j] = c_i_.size();
    work_estimate_ += 10;
    return true;
  }

  void ensure_row_space(i_t i, i_t needed)
  {
    if (row_end_[i] + needed <= row_max_[i]) { return; }
    const i_t r_start = row_start_[i];
    const i_t r_end   = row_end_[i];
    i_t current_size  = r_end - r_start;
    unused_row_nz_ += current_size;
    i_t new_start = r_j_.size();
    for (i_t p = r_start; p < r_end; p++) {
      r_j_.push_back(r_j_[p]);
    }
    work_estimate_ += (r_end - r_start);
    row_start_[i] = new_start;
    row_end_[i]   = r_j_.size();
    i_t extra     = std::max(current_size, needed);
    for (i_t k = 0; k < extra; k++) {
      r_j_.push_back(kNone);
    }
    work_estimate_ += extra;
    row_max_[i] = r_j_.size();
    work_estimate_ += 9;
  }

  // Compute the symmetric degree of each node: number of off-diagonal nonzeros
  // in the lower triangle that touch node j (entries in column j with row > j,
  // plus entries in columns k < j at row j).
  std::vector<i_t> compute_degree(const csc_matrix_t<i_t, f_t>& A)
  {
    std::vector<i_t> degree(A.n, 0);
    for (i_t j = 0; j < A.n; j++) {
      for (i_t p = A.col_start[j]; p < A.col_start[j + 1]; p++) {
        const i_t i = A.i[p];
        if (i == j) { continue; }  // skip diagonal
        // Off-diagonal entry (i, j) with i > j: contributes to degree of both i and j
        degree[j]++;
        degree[i]++;
      }
    }
    work_estimate_ += 3 * A.n + 2 * Bnz_;
    return degree;
  }

  i_t n_;
  i_t Bnz_;
  f_t work_estimate_;

  // Column representation of lower triangle (off-diagonal: rows > col)
  std::vector<i_t> col_start_;
  std::vector<i_t> col_end_;
  std::vector<i_t> col_max_;
  std::vector<i_t> c_i_;
  std::vector<f_t> c_x_;

  // Row representation (index only): row i stores columns j < i with entry (i, j)
  std::vector<i_t> row_start_;
  std::vector<i_t> row_end_;
  std::vector<i_t> row_max_;
  std::vector<i_t> r_j_;

  // Diagonal entries
  std::vector<f_t> diag_;

  // Dense workspaces for pivot column
  std::vector<f_t> pivot_col_val_;
  std::vector<char> pivot_col_mark_;
  std::vector<i_t> pivot_col_index_;

  // Single degree structure (symmetric: row degree == col degree)
  nonzero_counts_t<i_t, f_t> counts_;

  i_t unused_col_nz_;
  i_t unused_row_nz_;
};

}  // namespace

template <typename i_t, typename f_t>
i_t right_looking_ldlt(const csc_matrix_t<i_t, f_t>& A,
                       const simplex_solver_settings_t<i_t, f_t>& settings,
                       f_t pivot_tol,
                       f_t start_time,
                       std::vector<i_t>& perm,
                       csc_matrix_t<i_t, f_t>& L,
                       std::vector<f_t>& D,
                       f_t& work_estimate)
{
  raft::common::nvtx::range scope("LU::right_looking_ldlt");
  const i_t n = A.n;
  assert(A.m == n);

  symmetric_trailing_matrix_t<i_t, f_t> trailing_matrix(A);

  perm.resize(n);
  std::fill(perm.begin(), perm.end(), -1);
  D.clear();
  D.reserve(n);

  L.m = n;
  L.n = n;
  L.col_start.resize(n + 1);
  L.i.clear();
  L.x.clear();

  // perminv[original_index] = elimination_order
  std::vector<i_t> perminv(n, -1);

  i_t Lnz                = 0;
  constexpr f_t drop_tol = 1e-14;

  work_estimate += trailing_matrix.record_and_clear_work_estimate_();

  i_t pivots = 0;
  for (i_t k = 0; k < n; ++k) {
    if (settings.concurrent_halt != nullptr && *settings.concurrent_halt == 1) {
      return CONCURRENT_HALT_RETURN;
    }
    if (toc(start_time) > settings.time_limit) { return TIME_LIMIT_RETURN; }

    // Find symmetric pivot
    i_t pivot_p   = -1;
    f_t pivot_val = 0;
    trailing_matrix.symmetric_markowitz_search(pivot_tol, pivot_p, pivot_val);

    if (pivot_p == -1) { break; }  // No acceptable pivot found (remaining diagonals are zero/tiny)

    // Check for indefiniteness: a negative pivot means the matrix is not PSD
    if (pivot_val < 0) { return INDEFINITE_MATRIX_RETURN; }

    // Record permutation
    perm[k]          = pivot_p;
    perminv[pivot_p] = k;
    D.push_back(pivot_val);
    pivots++;

    // L(:, k) = entries in column pivot_p / pivot_val
    L.col_start[k] = Lnz;
    // Unit diagonal: L(pivot_p, k) = 1 (implicit, stored for triangular solve compatibility)
    L.i.push_back(pivot_p);
    L.x.push_back(1.0);
    Lnz++;
    trailing_matrix.extract_column(pivot_p, pivot_val, L, Lnz);

    // Symmetric Schur complement: A <- A - d * l * l^T
    trailing_matrix.symmetric_schur_complement(pivot_p, drop_tol, pivot_val);

    // Remove pivot from trailing matrix
    trailing_matrix.remove_pivot(pivot_p);

    trailing_matrix.garbage_collect();

    work_estimate += trailing_matrix.record_and_clear_work_estimate_();
  }

  // Finalize L
  L.col_start[pivots] = Lnz;
  // Fill remaining col_start entries for any unpivoted columns
  for (i_t k = pivots + 1; k <= n; k++) {
    L.col_start[k] = Lnz;
  }

  // Complete the permutation for unpivoted nodes
  {
    i_t next = pivots;
    for (i_t i = 0; i < n; i++) {
      if (perminv[i] == -1) {
        perminv[i] = next;
        if (next < n) { perm[next] = i; }
        next++;
      }
    }
  }

  // Remap L row indices from original to permuted order
  for (i_t p = 0; p < Lnz; p++) {
    L.i[p] = perminv[L.i[p]];
  }
  work_estimate += 2 * Lnz;

  // Resize L to actual rank
  L.n      = pivots;
  L.nz_max = Lnz;

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

template int right_looking_lu_row_permutation_only<int, double>(
  const csc_matrix_t<int, double>& A,
  const simplex_solver_settings_t<int, double>& settings,
  double tol,
  double start_time,
  std::vector<int>& q,
  std::vector<int>& pinv);

template int right_looking_ldlt<int, double>(const csc_matrix_t<int, double>& A,
                                             const simplex_solver_settings_t<int, double>& settings,
                                             double pivot_tol,
                                             double start_time,
                                             std::vector<int>& perm,
                                             csc_matrix_t<int, double>& L,
                                             std::vector<double>& D,
                                             double& work_estimate);
#endif

}  // namespace cuopt::mathematical_optimization::simplex
