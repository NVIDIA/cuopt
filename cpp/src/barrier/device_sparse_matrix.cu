/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <barrier/device_sparse_matrix.cuh>
#include <barrier/pinned_host_allocator.hpp>

#include <linear_algebra/sparse_matrix.hpp>

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

// One block per CSC column; each nonzero claims its CSR slot through its row's atomic cursor.
template <typename i_t, typename f_t>
__global__ void csc_to_csr_scatter_kernel(i_t n_cols,
                                          const i_t* __restrict__ col_start,
                                          const i_t* __restrict__ row_ind,
                                          const f_t* __restrict__ csc_val,
                                          i_t* __restrict__ next_pos,
                                          i_t* __restrict__ col_ind_out,
                                          f_t* __restrict__ val_out)
{
  const i_t col = static_cast<i_t>(blockIdx.x);
  if (col >= n_cols) { return; }
  const i_t thread_id = static_cast<i_t>(threadIdx.x);
  const i_t block_dim = static_cast<i_t>(blockDim.x);
  const i_t col_end   = col_start[col + 1];
  for (i_t p = col_start[col] + thread_id; p < col_end; p += block_dim) {
    const i_t q    = atomicAdd(next_pos + row_ind[p], i_t(1));
    col_ind_out[q] = col;
    val_out[q]     = csc_val[p];
  }
}

// Device CSC -> CSR on raw arrays. Doubles as a CSC transpose: CSR(A) and CSC(A^T) hold the
// same three arrays, so only the dimensions the caller records differ.
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
                          cuda::stream_ref stream)
{
  static_assert(std::is_signed_v<i_t>);

  if (nz == 0) {
    // Empty matrix: offsets all zero; indices/values unused.
    RAFT_CUDA_TRY(cudaMemsetAsync(out_offsets, 0, sizeof(i_t) * (m + 1), stream.get()));
    return;
  }

  auto exec = rmm::exec_policy(stream);

  // Per-row nnz from the CSC row indices (one atomic add per nonzero).
  rmm::device_uvector<i_t> row_counts(m, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(row_counts.data(), 0, sizeof(i_t) * m, stream.get()));

  thrust::for_each(exec,
                   thrust::make_counting_iterator<i_t>(0),
                   thrust::make_counting_iterator<i_t>(nz),
                   [row_ind, counts = row_counts.data()] __device__(i_t p) {
                     atomicAdd(counts + row_ind[p], i_t(1));
                   });

  // Row pointers: exclusive prefix sum of row_counts; out_offsets[m] = nz.
  rmm::device_buffer scan_tmp;
  std::size_t scan_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
    nullptr, scan_bytes, row_counts.data(), out_offsets, m, stream.get());
  scan_tmp.resize(scan_bytes, stream);
  cub::DeviceScan::ExclusiveSum(
    scan_tmp.data(), scan_bytes, row_counts.data(), out_offsets, m, stream.get());

  RAFT_CUDA_TRY(
    cudaMemcpyAsync(out_offsets + m, &nz, sizeof(i_t), cudaMemcpyHostToDevice, stream.get()));

  // Scatter every nonzero into its row's segment.
  rmm::device_uvector<i_t> next_pos(m, stream);
  raft::copy(next_pos.data(), out_offsets, m, stream);

  rmm::device_uvector<i_t> indices_unsorted(nz, stream);
  rmm::device_uvector<f_t> values_unsorted(nz, stream);
  constexpr int scatter_block_size = 256;
  csc_to_csr_scatter_kernel<i_t, f_t>
    <<<static_cast<unsigned int>(n), scatter_block_size, 0, stream.get()>>>(n,
                                                                            col_start,
                                                                            row_ind,
                                                                            csc_val,
                                                                            next_pos.data(),
                                                                            indices_unsorted.data(),
                                                                            values_unsorted.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Sort each segment by index; column ids are unique per row, so the result is deterministic.
  rmm::device_buffer sort_tmp;
  std::size_t sort_bytes = 0;
  cub::DeviceSegmentedSort::SortPairs(nullptr,
                                      sort_bytes,
                                      indices_unsorted.data(),
                                      out_indices,
                                      values_unsorted.data(),
                                      out_values,
                                      nz,
                                      m,
                                      out_offsets,
                                      out_offsets + 1,
                                      stream.get());
  sort_tmp.resize(sort_bytes, stream);
  cub::DeviceSegmentedSort::SortPairs(sort_tmp.data(),
                                      sort_bytes,
                                      indices_unsorted.data(),
                                      out_indices,
                                      values_unsorted.data(),
                                      out_values,
                                      nz,
                                      m,
                                      out_offsets,
                                      out_offsets + 1,
                                      stream.get());
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
                                                cuda::stream_ref stream);

#endif

}  // namespace cuopt::mathematical_optimization::barrier
