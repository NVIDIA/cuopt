/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

namespace {

// 64-bit integer mixer (SplitMix64). Used as the building block for the
// cousin filter's per-slot independent hash family.
inline uint64_t splitmix64_mix(uint64_t x)
{
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x = x ^ (x >> 31);
  return x;
}

inline uint64_t hash64_with_seed(uint64_t value, uint64_t seed)
{
  return splitmix64_mix(value ^ (seed * 0xbf58476d1ce4e5b9ULL + 0x9e3779b97f4a7c15ULL));
}

}  // namespace

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::add_cut(cut_type_t cut_type, const inequality_t<i_t, f_t>& cut)
{
  // TODO: Add fast duplicate check and only add if the cut is not already in the pool

  for (i_t p = 0; p < cut.size(); p++) {
    const i_t j = cut.index(p);
    if (j >= original_vars_) {
      settings_.log.printf(
        "Cut has variable %d that is greater than original_vars_ %d\n", j, original_vars_);
      return;
    }
  }

  inequality_t<i_t, f_t> cut_squeezed;
  cut.squeeze(cut_squeezed);
  if (cut_squeezed.size() == 0) {
    settings_.log.printf("Cut has no coefficients\n");
    return;
  }

  cut_storage_.append_row(cut_squeezed.vector);
  rhs_storage_.push_back(cut_squeezed.rhs);
  cut_type_.push_back(cut_type);
  cut_age_.push_back(0);
}

template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_distance(i_t row,
                                       const std::vector<f_t>& x,
                                       f_t& cut_violation,
                                       f_t& cut_norm)
{
  const i_t row_start = cut_storage_.row_start[row];
  const i_t row_end   = cut_storage_.row_start[row + 1];
  f_t cut_x           = 0.0;
  f_t dot             = 0.0;
  for (i_t p = row_start; p < row_end; p++) {
    const i_t j         = cut_storage_.j[p];
    const f_t cut_coeff = cut_storage_.x[p];
    cut_x += cut_coeff * x[j];
    dot += cut_coeff * cut_coeff;
  }
  cut_violation      = rhs_storage_[row] - cut_x;
  cut_norm           = std::sqrt(dot);
  const f_t distance = cut_violation / cut_norm;
  return distance;
}

template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_density(i_t row)
{
  const i_t row_start     = cut_storage_.row_start[row];
  const i_t row_end       = cut_storage_.row_start[row + 1];
  const i_t cut_nz        = row_end - row_start;
  const i_t original_vars = original_vars_;
  return static_cast<f_t>(cut_nz) / original_vars;
}

template <typename i_t, typename f_t>
f_t cut_pool_t<i_t, f_t>::cut_orthogonality(i_t i, i_t j)
{
  const i_t i_start = cut_storage_.row_start[i];
  const i_t i_end   = cut_storage_.row_start[i + 1];
  const i_t i_nz    = i_end - i_start;
  const i_t j_start = cut_storage_.row_start[j];
  const i_t j_end   = cut_storage_.row_start[j + 1];
  const i_t j_nz    = j_end - j_start;

  f_t dot = sparse_dot(cut_storage_.j.data() + i_start,
                       cut_storage_.x.data() + i_start,
                       i_nz,
                       cut_storage_.j.data() + j_start,
                       cut_storage_.x.data() + j_start,
                       j_nz);

  f_t norm_i = cut_norms_[i];
  f_t norm_j = cut_norms_[j];
  return 1.0 - std::abs(dot) / (norm_i * norm_j);
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::check_for_duplicate_cuts()
{
  const i_t m = cut_storage_.m;

  constexpr f_t duplicate_tolerance = 1e-10;
  std::vector<f_t> divisors(m, 0.0);
  std::vector<i_t> sets(m, 0);
  csc_matrix_t<i_t, f_t> cut_storage_csc(0, 0, 1);
  cut_storage_.to_compressed_col(cut_storage_csc);
  const i_t n        = cut_storage_csc.n;
  const i_t sentinel = std::numeric_limits<i_t>::max();

  // Algorithm from Finding Duplicate Rows in a Linear Programming Model
  // by J. A. Tomlin and J.S. Welch
  // Operations Research Letters Volume 5, Number 1, June 1986.
  //
  // Preserve the legacy partition refinement and row-ordered deletion semantics, but group
  // entries by their current set in compressed storage. This avoids scanning unrelated later
  // entries without changing the first matching partner or the resulting removal mask.
  i_t new_set                        = 1;
  i_t remaining_potential_duplicates = m;
  compressed_set_groups_t<i_t> set_groups;
  for (i_t j = 0; j < n; j++) {
    i_t r0        = -1;
    i_t new_rows  = 0;
    i_t new_set_0 = new_set;
    new_set++;
    const i_t col_start = cut_storage_csc.col_start[j];
    const i_t col_end   = cut_storage_csc.col_start[j + 1];

    set_groups.build_with_entry_indices(col_start, col_end, new_set_0, cut_storage_csc.i, sets);

    for (i_t p = col_start; p < col_end; p++) {
      const i_t r    = cut_storage_csc.i[p];
      const f_t a_rj = cut_storage_csc.x[p];
      const f_t f_r  = divisors[r];
      if (sets[r] == 0) {
        r0          = r;  // To enable us to find this new set later.
        sets[r]     = new_set_0;
        divisors[r] = a_rj;
        new_rows++;
      } else if (sets[r] < new_set_0) {
        const i_t old_set = sets[r];
        bool matched      = false;
        // Loop over all indices a_ij with i > r where i is in the same set as r.
        for (const i_t q : set_groups.entries_after(old_set, p)) {
          const i_t i    = cut_storage_csc.i[q];
          const f_t a_ij = cut_storage_csc.x[q];
          if (sets[i] != old_set) { continue; }
          const f_t f_i = divisors[i];
          const f_t val = (a_rj / f_r) * (f_i / a_ij);
          if (val >= 1.0 - duplicate_tolerance && val <= 1.0 + duplicate_tolerance) {
            sets[r] = new_set;
            sets[i] = new_set;
            matched = true;
            break;
          }
        }
        if (matched) {
          new_set++;
        } else {
          sets[r] = sentinel;
          remaining_potential_duplicates--;
          if (remaining_potential_duplicates == 0) { break; }
        }
      }
    }
    if (remaining_potential_duplicates == 0) { break; }
    if (new_rows == 1) {
      sets[r0] = sentinel;
      remaining_potential_duplicates--;
      if (remaining_potential_duplicates == 0) { break; }
    }
  }

  // The cuts are stored in the form: sum_j d_ij x_j >= rhs_i.
  // We now look for cuts that are duplicates of each other and remove them.
  set_groups.build(0, m, new_set, sets);

  std::vector<i_t> cuts_to_remove(m, 0);
  i_t num_cuts_to_remove = 0;
  for (i_t r = 0; r < m; r++) {
    const i_t set_r = sets[r];
    if (set_r <= 0 || set_r >= sentinel || cuts_to_remove[r] != 0) { continue; }
    // This cut has a duplicate. The set members are in row order, preserving the legacy
    // strongest-cut selection order without scanning unrelated rows.
    for (const i_t i : set_groups.entries_after(set_r, r)) {
      const f_t f_r     = divisors[r];
      const f_t f_i     = divisors[i];
      const f_t theta_r = rhs_storage_[r] / f_r;
      const f_t theta_i = rhs_storage_[i] / f_i;
      if (f_r > 0.0 && f_i > 0.0) {
        // We have sum_j d_rj / f_r x_j >= rhs_r / f_r = theta_r
        //    and  sum_j d_ij / f_i x_j >= rhs_i / f_i = theta_i.
        if (theta_r <= theta_i) {
          // Cut i is either the same or stronger than cut r.
          if (cuts_to_remove[r] == 0) { num_cuts_to_remove++; }
          cuts_to_remove[r] = 1;  // Remove row r.
        } else {
          // theta_r > theta_i, so cut r is strictly stronger than cut i.
          if (cuts_to_remove[i] == 0) { num_cuts_to_remove++; }
          cuts_to_remove[i] = 1;  // Remove row i.
        }
      } else if (f_r < 0.0 && f_i < 0.0) {
        // Dividing by a negative divisor reverses the inequality:
        // sum_j d_rj / f_r x_j <= rhs_r / f_r = theta_r
        //    and  sum_j d_ij / f_i x_j <= rhs_i / f_i = theta_i.
        if (theta_r >= theta_i) {
          // Cut i is either the same or stronger than cut r.
          if (cuts_to_remove[r] == 0) { num_cuts_to_remove++; }
          cuts_to_remove[r] = 1;  // Remove row r.
        } else {
          // theta_r < theta_i, so cut r is strictly stronger than cut i.
          if (cuts_to_remove[i] == 0) { num_cuts_to_remove++; }
          cuts_to_remove[i] = 1;  // Remove row i.
        }
      }
    }
  }

  if (num_cuts_to_remove > 0) {
    settings_.log.debug("Removing %d duplicate cuts\n", num_cuts_to_remove);
    csr_matrix_t<i_t, f_t> new_cut_storage(0, 0, 0);
    cut_storage_.remove_rows(cuts_to_remove, new_cut_storage);
    cut_storage_ = new_cut_storage;
    i_t write    = 0;
    for (i_t i = 0; i < m; i++) {
      if (cuts_to_remove[i] == 0) {
        rhs_storage_[write] = rhs_storage_[i];
        cut_type_[write]    = cut_type_[i];
        cut_age_[write]     = cut_age_[i];
        write++;
      }
    }
    rhs_storage_.resize(write);
    cut_type_.resize(write);
    cut_age_.resize(write);
  }
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::score_cuts(std::vector<f_t>& x_relax)
{
  check_for_duplicate_cuts();
  cut_distances_.resize(cut_storage_.m, 0.0);
  cut_norms_.resize(cut_storage_.m, 0.0);

  const bool verbose = false;
  for (i_t i = 0; i < cut_storage_.m; i++) {
    f_t violation;
    f_t cut_dist      = cut_distance(i, x_relax, violation, cut_norms_[i]);
    cut_distances_[i] = cut_dist <= min_cut_distance_ ? 0.0 : cut_dist;
    if (verbose) {
      settings_.log.printf("Cut %d type %d distance %+e violation %+e cut_norm %e\n",
                           i,
                           static_cast<int>(cut_type_[i]),
                           cut_distances_[i],
                           violation,
                           cut_norms_[i]);
    }
  }

  std::vector<i_t> sorted_indices;
  best_score_last_permutation(cut_distances_, sorted_indices);

  const i_t max_cuts          = 2000;
  const f_t min_orthogonality = settings_.cut_min_orthogonality;
  best_cuts_.reserve(std::min(max_cuts, cut_storage_.m));
  best_cuts_.clear();
  scored_cuts_ = 0;

  if (!sorted_indices.empty()) {
    const i_t i = sorted_indices.back();
    sorted_indices.pop_back();
    best_cuts_.push_back(i);
    scored_cuts_++;
  }

  while (scored_cuts_ < max_cuts && !sorted_indices.empty()) {
    const i_t i = sorted_indices.back();
    sorted_indices.pop_back();

    if (cut_distances_[i] <= min_cut_distance_) { break; }

    f_t cut_ortho            = 1.0;
    const i_t best_cuts_size = best_cuts_.size();
    for (i_t k = 0; k < best_cuts_size; k++) {
      const i_t j = best_cuts_[k];
      cut_ortho   = std::min(cut_ortho, cut_orthogonality(i, j));
    }
    if (cut_ortho >= min_orthogonality) {
      best_cuts_.push_back(i);
      scored_cuts_++;
    }
  }
}

template <typename i_t, typename f_t>
i_t cut_pool_t<i_t, f_t>::get_best_cuts(csr_matrix_t<i_t, f_t>& best_cuts,
                                        std::vector<f_t>& best_rhs,
                                        std::vector<cut_type_t>& best_cut_types)
{
  best_cuts.m = 0;
  best_cuts.n = original_vars_;
  best_cuts.row_start.clear();
  best_cuts.j.clear();
  best_cuts.x.clear();
  best_cuts.row_start.reserve(scored_cuts_ + 1);
  best_cuts.row_start.push_back(0);
  best_rhs.clear();
  best_rhs.reserve(scored_cuts_);
  best_cut_types.clear();
  best_cut_types.reserve(scored_cuts_);

  for (i_t i : best_cuts_) {
    if (cut_distances_[i] <= min_cut_distance_) { continue; }
    sparse_vector_t<i_t, f_t> cut(cut_storage_, i);
    cut.negate();
    best_cuts.append_row(cut);
    best_rhs.push_back(-rhs_storage_[i]);
    best_cut_types.push_back(cut_type_[i]);
  }

  age_cuts();

  return static_cast<i_t>(best_rhs.size());
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::age_cuts()
{
  for (i_t i = 0; i < cut_age_.size(); i++) {
    cut_age_[i]++;
  }
}

template <typename i_t, typename f_t>
void cut_pool_t<i_t, f_t>::drop_cuts()
{
  // TODO: Implement this
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class cut_pool_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
