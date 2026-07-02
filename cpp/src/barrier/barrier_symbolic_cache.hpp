/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <barrier/barrier_factorization_sparsity_hash.hpp>
#include <barrier/cusparse_info.hpp>
#include <barrier/device_sparse_matrix.cuh>

#include <raft/core/handle.hpp>

#include <cstdint>
#include <memory>

namespace cuopt::linear_programming::dual_simplex {

using barrier_sparsity_hash_t = std::uint64_t;

template <typename i_t, typename f_t>
class sparse_cholesky_cudss_t;

/**
 * @brief Cached cuDSS symbolic state and GPU buffers for hash-gated barrier reuse.
 *
 * Holds reordering + symbolic factorization in @p chol, the sparsity hash of the matrix
 * that was analyzed, and path-specific GPU workspace (augmented KKT or ADAT + cuSPARSE).
 */
template <typename i_t, typename f_t>
struct barrier_symbolic_cache_t {
  std::shared_ptr<sparse_cholesky_cudss_t<i_t, f_t>> chol;
  barrier_sparsity_hash_t sparsity_hash{0};
  raft::handle_t const* handle_ptr{nullptr};
  bool use_augmented{false};
  bool valid{false};

  // --- Augmented KKT (use_augmented == true) ---
  device_csr_matrix_t<i_t, f_t> device_augmented;
  rmm::device_uvector<i_t> d_augmented_diagonal_indices_;

  // --- ADAT (use_augmented == false) ---
  device_csr_matrix_t<i_t, f_t> device_ADAT;
  device_csc_matrix_t<i_t, f_t> device_AD;
  device_csr_matrix_t<i_t, f_t> device_A;
  rmm::device_uvector<f_t> d_original_A_values;
  rmm::device_uvector<f_t> device_A_x_values;
  std::unique_ptr<cusparse_info_t<i_t, f_t>> cusparse_info;

  explicit barrier_symbolic_cache_t(rmm::cuda_stream_view stream)
    : device_augmented(stream),
      d_augmented_diagonal_indices_(0, stream),
      device_ADAT(stream),
      device_AD(stream),
      device_A(stream),
      d_original_A_values(0, stream),
      device_A_x_values(0, stream)
  {
  }

  void clear()
  {
    chol.reset();
    sparsity_hash = 0;
    handle_ptr    = nullptr;
    use_augmented = false;
    valid         = false;
    cusparse_info.reset();
  }

  [[nodiscard]] bool matches_reuse(barrier_sparsity_hash_t hash,
                                   bool augmented,
                                   raft::handle_t const* handle) const
  {
    return valid && handle != nullptr && handle_ptr == handle && use_augmented == augmented &&
           sparsity_hash == hash;
  }
};

}  // namespace cuopt::linear_programming::dual_simplex
