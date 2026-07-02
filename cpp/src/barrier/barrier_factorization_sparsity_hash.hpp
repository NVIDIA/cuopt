/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <barrier/device_sparse_matrix.cuh>

#include <dual_simplex/sparse_matrix.hpp>

#include <cstdint>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

using barrier_sparsity_hash_t = std::uint64_t;

/// FNV-1a style mix for incremental hashing.
inline barrier_sparsity_hash_t barrier_hash_combine(barrier_sparsity_hash_t h, std::uint64_t value)
{
  constexpr barrier_sparsity_hash_t kPrime = 1099511628211ULL;
  h ^= value;
  h *= kPrime;
  return h;
}

inline barrier_sparsity_hash_t barrier_hash_u64(std::uint64_t value)
{
  return barrier_hash_combine(1469598103934665603ULL, value);
}

/**
 * @brief Hash CSR sparsity (row_start + col indices); numeric values are ignored.
 */
template <typename i_t>
barrier_sparsity_hash_t hash_host_csr_sparsity_pattern(i_t num_rows,
                                                       const std::vector<i_t>& row_start,
                                                       const std::vector<i_t>& col_indices)
{
  barrier_sparsity_hash_t h = barrier_hash_u64(static_cast<std::uint64_t>(num_rows));
  h                         = barrier_hash_combine(h, static_cast<std::uint64_t>(col_indices.size()));
  for (i_t k = 0; k <= num_rows; ++k) {
    h = barrier_hash_combine(h, static_cast<std::uint64_t>(row_start[static_cast<size_t>(k)]));
  }
  for (i_t col : col_indices) {
    h = barrier_hash_combine(h, static_cast<std::uint64_t>(col));
  }
  return h;
}

/**
 * @brief Hash the sparsity pattern of the augmented KKT matrix passed to cuDSS (host CSR).
 *
 * Must match the index layout produced by iteration_data_t::form_augmented(true).
 */
template <typename i_t, typename f_t>
barrier_sparsity_hash_t hash_augmented_kkt_sparsity(const csc_matrix_t<i_t, f_t>& A,
                                                    const csc_matrix_t<i_t, f_t>& AT,
                                                    const csc_matrix_t<i_t, f_t>& Q)
{
  const i_t n    = A.n;
  const i_t m    = A.m;
  const i_t size = n + m;

  std::vector<i_t> row_start(static_cast<size_t>(size + 1), 0);
  std::vector<i_t> col_indices;
  col_indices.reserve(static_cast<size_t>(2) * static_cast<size_t>(A.col_start[n]) +
                      static_cast<size_t>(n + m) +
                      (Q.n > 0 ? static_cast<size_t>(Q.col_start[n]) : 0));

  i_t q = 0;
  for (i_t i = 0; i < n; ++i) {
    row_start[static_cast<size_t>(i)] = q;
    if (Q.n == 0) {
      col_indices.push_back(i);
      ++q;
    } else {
      const i_t q_col_beg = Q.col_start[i];
      const i_t q_col_end = Q.col_start[i + 1];
      bool has_diagonal   = false;
      for (i_t p = q_col_beg; p < q_col_end; ++p) {
        col_indices.push_back(Q.i[p]);
        ++q;
        if (Q.i[p] == i) { has_diagonal = true; }
      }
      if (!has_diagonal) {
        col_indices.push_back(i);
        ++q;
      }
    }
    const i_t col_beg = A.col_start[i];
    const i_t col_end = A.col_start[i + 1];
    for (i_t p = col_beg; p < col_end; ++p) {
      col_indices.push_back(A.i[p] + n);
      ++q;
    }
  }

  for (i_t k = n; k < n + m; ++k) {
    row_start[static_cast<size_t>(k)] = q;
    const i_t l                       = k - n;
    const i_t col_beg                 = AT.col_start[l];
    const i_t col_end                 = AT.col_start[l + 1];
    for (i_t p = col_beg; p < col_end; ++p) {
      col_indices.push_back(AT.i[p]);
      ++q;
    }
    col_indices.push_back(k);
    ++q;
  }
  row_start[static_cast<size_t>(size)] = q;

  return hash_host_csr_sparsity_pattern(size, row_start, col_indices);
}

/**
 * @brief Hash CSR sparsity from a device matrix (copies row/col indices to host).
 */
template <typename i_t, typename f_t>
barrier_sparsity_hash_t hash_device_csr_sparsity_pattern(
  device_csr_matrix_t<i_t, f_t>& mat, rmm::cuda_stream_view stream);

}  // namespace cuopt::linear_programming::dual_simplex
