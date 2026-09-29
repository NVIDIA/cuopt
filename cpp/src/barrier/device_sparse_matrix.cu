/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <barrier/device_sparse_matrix.cuh>
#include <barrier/pinned_host_allocator.hpp>

#include <linear_algebra/sparse_matrix.hpp>

#include <raft/sparse/linalg/transpose.cuh>

// This translation unit provides out-of-line definitions and explicit
// instantiations of shared sparse-matrix templates (csc_matrix_t,
// matrix_transpose_vector_multiply) specialized with barrier's
// PinnedHostAllocator. They must live in the mathematical_optimization namespace
// (where the templates are declared), even though the file resides under barrier/.
namespace cuopt::mathematical_optimization {

using cuopt::mathematical_optimization::barrier::PinnedHostAllocator;

template <typename i_t, typename f_t>
template <typename Allocator>
void csc_matrix_t<i_t, f_t>::scale_columns(const std::vector<f_t, Allocator>& scale)
{
  const i_t n = this->n;
  assert(scale.size() == n);
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = this->col_start[j];
    const i_t col_end   = this->col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      this->x[p] *= scale[j];
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

// NOTE: matrix_vector_multiply is now templated on VectorX and VectorY.
// Since it's defined inline in the header, no explicit instantiation is needed here.

template int matrix_transpose_vector_multiply<int,
                                              double,
                                              PinnedHostAllocator<double>,
                                              PinnedHostAllocator<double>>(
  const csc_matrix_t<int, double>& A,
  double alpha,
  const std::vector<double, PinnedHostAllocator<double>>& x,
  double beta,
  std::vector<double, PinnedHostAllocator<double>>& y);

template int
matrix_transpose_vector_multiply<int, double, PinnedHostAllocator<double>, std::allocator<double>>(
  const csc_matrix_t<int, double>& A,
  double alpha,
  const std::vector<double, PinnedHostAllocator<double>>& x,
  double beta,
  std::vector<double, std::allocator<double>>& y);

template int
matrix_transpose_vector_multiply<int, double, std::allocator<double>, PinnedHostAllocator<double>>(
  const csc_matrix_t<int, double>& A,
  double alpha,
  const std::vector<double, std::allocator<double>>& x,
  double beta,
  std::vector<double, PinnedHostAllocator<double>>& y);

template void csc_matrix_t<int, double>::scale_columns<std::allocator<double>>(
  const std::vector<double, std::allocator<double>>& scale);
template void csc_matrix_t<int, double>::scale_columns<PinnedHostAllocator<double>>(
  const std::vector<double, PinnedHostAllocator<double>>& scale);

#endif

}  // namespace cuopt::mathematical_optimization

namespace cuopt::mathematical_optimization::barrier {

// Device CSC -> CSR. CSC(A) is exactly CSR(A^T), so transposing it yields CSR(A).
template <typename i_t, typename f_t>
void csc_to_csr_on_device(i_t m,
                          i_t n,
                          i_t nz,
                          const i_t* col_start,
                          const i_t* row_ind,
                          const f_t* csc_val,
                          i_t* out_offsets,
                          i_t* out_indices,
                          f_t* out_values,
                          const raft::handle_t* handle_ptr)
{
  static_assert(std::is_signed_v<i_t>);
  const auto stream = handle_ptr->get_stream();

  if (nz == 0) {
    // Empty matrix: offsets all zero; indices/values unused.
    RAFT_CUDA_TRY(cudaMemsetAsync(out_offsets, 0, sizeof(i_t) * (m + 1), stream.value()));
    return;
  }

  raft::sparse::linalg::csr_transpose(*handle_ptr,
                                      col_start,
                                      row_ind,
                                      csc_val,
                                      out_offsets,
                                      out_indices,
                                      out_values,
                                      n,
                                      m,
                                      nz,
                                      stream.value());
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template void csc_to_csr_on_device<int, double>(int m,
                                                int n,
                                                int nz,
                                                const int* col_start,
                                                const int* row_ind,
                                                const double* csc_values,
                                                int* out_offsets,
                                                int* out_indices,
                                                double* out_values,
                                                const raft::handle_t* handle_ptr);

#endif

}  // namespace cuopt::mathematical_optimization::barrier
