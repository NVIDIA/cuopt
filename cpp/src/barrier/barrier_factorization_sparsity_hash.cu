/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <barrier/barrier_factorization_sparsity_hash.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
barrier_sparsity_hash_t hash_device_csr_sparsity_pattern(
  device_csr_matrix_t<i_t, f_t>& mat, rmm::cuda_stream_view stream)
{
  const csr_matrix_t<i_t, f_t> host = mat.to_host(stream);
  return hash_host_csr_sparsity_pattern(host.m, host.row_start, host.j);
}

template barrier_sparsity_hash_t hash_device_csr_sparsity_pattern<int, float>(
  device_csr_matrix_t<int, float>&, rmm::cuda_stream_view);
template barrier_sparsity_hash_t hash_device_csr_sparsity_pattern<int, double>(
  device_csr_matrix_t<int, double>&, rmm::cuda_stream_view);

}  // namespace cuopt::linear_programming::dual_simplex
