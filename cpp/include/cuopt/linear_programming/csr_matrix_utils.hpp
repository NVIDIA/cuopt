/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace cuopt::linear_programming {

/**
 * @brief Compare two CSR matrices under row and column permutations (host-side).
 *
 * For each row i in 'this', finds the matching row row_perm[i] in 'other',
 * remaps this's column indices to other's space via var_perm, sorts both rows
 * by column, and compares values with epsilon tolerance.
 *
 * @param this_offsets  CSR row offsets for 'this' matrix
 * @param this_indices  CSR column indices for 'this' matrix
 * @param this_values   CSR values for 'this' matrix
 * @param other_offsets CSR row offsets for 'other' matrix
 * @param other_indices CSR column indices for 'other' matrix
 * @param other_values  CSR values for 'other' matrix
 * @param row_perm      Forward row permutation: row_perm[i] = matching row in 'other' for this row
 * i
 * @param var_perm      Forward col permutation: var_perm[j] = matching col in 'other' for this col
 * j
 * @param tolerance     Epsilon tolerance for value comparison
 * @return true if matrices are equivalent under the given permutations
 */
template <typename i_t, typename f_t>
inline bool csr_matrices_equivalent_with_permutation_host(const std::vector<i_t>& this_offsets,
                                                          const std::vector<i_t>& this_indices,
                                                          const std::vector<f_t>& this_values,
                                                          const std::vector<i_t>& other_offsets,
                                                          const std::vector<i_t>& other_indices,
                                                          const std::vector<f_t>& other_values,
                                                          const std::vector<i_t>& row_perm,
                                                          const std::vector<i_t>& var_perm,
                                                          f_t tolerance = 1e-9)
{
  i_t n_rows = static_cast<i_t>(this_offsets.size()) - 1;
  if (n_rows != static_cast<i_t>(other_offsets.size()) - 1) return false;
  if (this_values.size() != other_values.size()) return false;

  for (i_t i = 0; i < n_rows; ++i) {
    i_t other_row = row_perm[i];

    i_t this_row_len  = this_offsets[i + 1] - this_offsets[i];
    i_t other_row_len = other_offsets[other_row + 1] - other_offsets[other_row];
    if (this_row_len != other_row_len) return false;

    // Gather (mapped_col, val) pairs for this row, mapping cols to other's space
    std::vector<std::pair<i_t, f_t>> this_entries;
    for (i_t k = this_offsets[i]; k < this_offsets[i + 1]; ++k) {
      this_entries.emplace_back(var_perm[this_indices[k]], this_values[k]);
    }

    // Gather (col, val) pairs for matching other row (already in other's space)
    std::vector<std::pair<i_t, f_t>> other_entries;
    for (i_t k = other_offsets[other_row]; k < other_offsets[other_row + 1]; ++k) {
      other_entries.emplace_back(other_indices[k], other_values[k]);
    }

    // Sort both by column index
    auto cmp = [](const std::pair<i_t, f_t>& a, const std::pair<i_t, f_t>& b) {
      return a.first < b.first;
    };
    std::sort(this_entries.begin(), this_entries.end(), cmp);
    std::sort(other_entries.begin(), other_entries.end(), cmp);

    // Compare element-wise
    for (size_t k = 0; k < this_entries.size(); ++k) {
      if (this_entries[k].first != other_entries[k].first) return false;
      if (std::abs(this_entries[k].second - other_entries[k].second) > tolerance) return false;
    }
  }

  return true;
}

}  // namespace cuopt::linear_programming
