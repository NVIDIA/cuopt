/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <linear_algebra/sparse_matrix.hpp>
#include <math_optimization/types.hpp>

#include <cub/cub.cuh>
#include <cuda/stream>
#include <rmm/device_scalar.hpp>
#include <rmm/device_vector.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/cuda_helpers.cuh>

#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>

namespace cuopt::mathematical_optimization::barrier {

template <typename IndexType, typename ValueType>
class device_csr_matrix_t;

template <typename f_t>
struct sum_reduce_helper_t {
  rmm::device_buffer buffer_data;
  rmm::device_scalar<f_t> out;
  size_t buffer_size;

  sum_reduce_helper_t(cuda::stream_ref stream_view) : buffer_data(0, stream_view), out(stream_view)
  {
  }

  template <typename InputIteratorT, typename i_t>
  f_t sum(InputIteratorT input, i_t size, cuda::stream_ref stream_view)
  {
    buffer_size = 0;
    cub::DeviceReduce::Sum(nullptr, buffer_size, input, out.data(), size, stream_view.get());
    buffer_data.resize(buffer_size, stream_view);
    cub::DeviceReduce::Sum(
      buffer_data.data(), buffer_size, input, out.data(), size, stream_view.get());
    return out.value(stream_view);
  }
};

template <typename f_t>
struct transform_reduce_helper_t {
  rmm::device_buffer buffer_data;
  rmm::device_scalar<f_t> out;
  size_t buffer_size;

  transform_reduce_helper_t(cuda::stream_ref stream_view)
    : buffer_data(0, stream_view), out(stream_view)
  {
  }

  template <typename InputIteratorT, typename ReductionOpT, typename TransformOpT, typename i_t>
  f_t transform_reduce(InputIteratorT input,
                       ReductionOpT reduce_op,
                       TransformOpT transform_op,
                       f_t init,
                       i_t size,
                       cuda::stream_ref stream_view)
  {
    cub::DeviceReduce::TransformReduce(nullptr,
                                       buffer_size,
                                       input,
                                       out.data(),
                                       size,
                                       reduce_op,
                                       transform_op,
                                       init,
                                       stream_view.get());

    buffer_data.resize(buffer_size, stream_view);

    cub::DeviceReduce::TransformReduce(buffer_data.data(),
                                       buffer_size,
                                       input,
                                       out.data(),
                                       size,
                                       reduce_op,
                                       transform_op,
                                       init,
                                       stream_view.get());

    return out.value(stream_view);
  }
};

template <typename f_t>
struct f2_t {
  f_t a;
  f_t b;
};

template <typename f_t>
struct f2_min_t {
  HD f2_t<f_t> operator()(const f2_t<f_t>& lhs, const f2_t<f_t>& rhs) const
  {
    return f2_t<f_t>{cuda::std::min(lhs.a, rhs.a), cuda::std::min(lhs.b, rhs.b)};
  }
};

template <typename f_t>
struct transform_reduce_pair_helper_t {
  rmm::device_buffer buffer_data;
  rmm::device_scalar<f2_t<f_t>> out;
  size_t buffer_size;

  transform_reduce_pair_helper_t(cuda::stream_ref stream_view)
    : buffer_data(0, stream_view), out(stream_view)
  {
  }

  // TransformOpT must map each input element to an f2_t<f_t>{a, b} pair; the two
  // components are reduced independently (elementwise min) in a single kernel launch.
  template <typename InputIteratorT, typename TransformOpT, typename i_t>
  f2_t<f_t> transform_reduce(InputIteratorT input,
                             TransformOpT transform_op,
                             f2_t<f_t> init,
                             i_t size,
                             cuda::stream_ref stream_view)
  {
    f2_min_t<f_t> reduce_op{};
    cub::DeviceReduce::TransformReduce(nullptr,
                                       buffer_size,
                                       input,
                                       out.data(),
                                       size,
                                       reduce_op,
                                       transform_op,
                                       init,
                                       stream_view.get());

    buffer_data.resize(buffer_size, stream_view);

    cub::DeviceReduce::TransformReduce(buffer_data.data(),
                                       buffer_size,
                                       input,
                                       out.data(),
                                       size,
                                       reduce_op,
                                       transform_op,
                                       init,
                                       stream_view.get());

    return out.value(stream_view);
  }
};

template <typename i_t, typename f_t>
struct csc_view_t {
  raft::device_span<i_t> col_start;
  raft::device_span<i_t> i;
  raft::device_span<f_t> x;
};

template <typename i_t, typename f_t>
class device_csc_matrix_t {
 public:
  device_csc_matrix_t(cuda::stream_ref stream)
    : col_start(0, stream), i(0, stream), x(0, stream), col_index(0, stream)
  {
  }

  device_csc_matrix_t(i_t rows, i_t cols, i_t nz, cuda::stream_ref stream)
    : m(rows),
      n(cols),
      nz_max(nz),
      col_start(cols + 1, stream),
      i(nz_max, stream),
      x(nz_max, stream),
      col_index(0, stream)
  {
  }

  device_csc_matrix_t(device_csc_matrix_t const& other)
    : nz_max(other.nz_max),
      m(other.m),
      n(other.n),
      col_start(other.col_start, other.col_start.stream()),
      i(other.i, other.i.stream()),
      x(other.x, other.x.stream()),
      col_index(other.col_index, other.col_index.stream())
  {
  }

  /** Move leaves the source empty; needed to hand a matrix over without a device-to-device copy. */
  device_csc_matrix_t(device_csc_matrix_t&&) = default;

  device_csc_matrix_t(const csc_matrix_t<i_t, f_t>& A, cuda::stream_ref stream)
    : m(A.m),
      n(A.n),
      nz_max(A.col_start[A.n]),
      col_start(A.col_start.size(), stream),
      i(A.i.size(), stream),
      x(A.x.size(), stream),
      col_index(0, stream)
  {
    col_start = cuopt::device_copy(A.col_start, stream);
    i         = cuopt::device_copy(A.i, stream);
    x         = cuopt::device_copy(A.x, stream);
  }

  void resize_to_nnz(i_t nnz, cuda::stream_ref stream)
  {
    col_start.resize(n + 1, stream);
    i.resize(nnz, stream);
    x.resize(nnz, stream);
    nz_max = nnz;
  }

  csc_matrix_t<i_t, f_t> to_host(cuda::stream_ref stream)
  {
    csc_matrix_t<i_t, f_t> A(m, n, nz_max);
    A.col_start = cuopt::host_copy(col_start, stream);
    A.i         = cuopt::host_copy(i, stream);
    A.x         = cuopt::host_copy(x, stream);
    return A;
  }

  void copy(const csc_matrix_t<i_t, f_t>& A, cuda::stream_ref stream)
  {
    m      = A.m;
    n      = A.n;
    nz_max = A.col_start[A.n];
    col_start.resize(A.col_start.size(), stream);
    raft::copy(col_start.data(), A.col_start.data(), A.col_start.size(), stream);
    i.resize(A.i.size(), stream);
    raft::copy(i.data(), A.i.data(), A.i.size(), stream);
    x.resize(A.x.size(), stream);
    raft::copy(x.data(), A.x.data(), A.x.size(), stream);
  }

  /** Reset to an empty (all-zero col_start, no nonzeros) matrix of the given shape. */
  void reset_empty(i_t rows, i_t cols, cuda::stream_ref stream)
  {
    m      = rows;
    n      = cols;
    nz_max = 0;
    resize_to_nnz(0, stream);
    thrust::fill(rmm::exec_policy(stream), col_start.begin(), col_start.end(), i_t(0));
  }

  /** Same semantics as csc_matrix_t::to_compressed_row, entirely on
   * device. */
  void to_compressed_row(device_csr_matrix_t<i_t, f_t>& Arow, cuda::stream_ref stream) const;

  /** Same semantics as csc_matrix_t::transpose, entirely on device. */
  void transpose(device_csc_matrix_t<i_t, f_t>& AT, cuda::stream_ref stream) const;

  /** Tag selecting the transpose constructor below. */
  struct transposed_t {};

  /** Construct as A^T, entirely on device. */
  device_csc_matrix_t(transposed_t, const device_csc_matrix_t& A, cuda::stream_ref stream)
    : col_start(0, stream), i(0, stream), x(0, stream), col_index(0, stream)
  {
    A.transpose(*this, stream);
  }

  void form_col_index(cuda::stream_ref stream)
  {
    col_index.resize(x.size(), stream);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(col_index.data(), 0, sizeof(i_t) * col_index.size(), stream.get()));

    // Scatter 1 when there is a col start in col_index
    if (col_start.size() > 2) {
      thrust::for_each(rmm::exec_policy(stream),
                       thrust::make_counting_iterator(i_t(1)),  // Skip the first 0
                       thrust::make_counting_iterator(
                         static_cast<i_t>(col_start.size() - 1)),  // Skip the end index
                       [span_col_start = cuopt::make_span(col_start),
                        span_col_index = cuopt::make_span(col_index)] __device__(i_t i) {
                         if (span_col_start[i] < span_col_index.size()) {
                           span_col_index[span_col_start[i]] = 1;
                         }
                       });
    }

    // Inclusive cumulative sum to have the corresponding column for each entry
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes{0};
    cub::DeviceScan::InclusiveSum(nullptr,
                                  temp_storage_bytes,
                                  col_index.data(),
                                  col_index.data(),
                                  col_index.size(),
                                  stream.get());
    d_temp_storage.resize(temp_storage_bytes, stream);
    cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  col_index.data(),
                                  col_index.data(),
                                  col_index.size(),
                                  stream.get());
    // Have to sync since InclusiveSum is being run on local data (d_temp_storage)
    stream.sync();
  }

  csc_view_t<i_t, f_t> view()
  {
    csc_view_t<i_t, f_t> v;
    v.col_start = cuopt::make_span(col_start);
    v.i         = cuopt::make_span(i);
    v.x         = cuopt::make_span(x);
    return v;
  }

  i_t nz_max;                          // maximum number of entries
  i_t m;                               // number of rows
  i_t n;                               // number of columns
  rmm::device_uvector<i_t> col_start;  // column pointers (size n + 1)
  rmm::device_uvector<i_t> i;          // row indices, size nz_max
  rmm::device_uvector<f_t> x;          // numerical values, size nz_max
  rmm::device_uvector<i_t> col_index;  // index of each column, only used for scale column
};

template <typename i_t, typename f_t>
class device_csr_matrix_t {
 public:
  device_csr_matrix_t(cuda::stream_ref stream) : row_start(0, stream), j(0, stream), x(0, stream) {}

  device_csr_matrix_t(i_t rows, i_t cols, i_t nz, cuda::stream_ref stream)
    : m(rows),
      n(cols),
      nz_max(nz),
      row_start(rows + 1, stream),
      j(nz_max, stream),
      x(nz_max, stream)
  {
  }

  device_csr_matrix_t(device_csr_matrix_t const& other)
    : nz_max(other.nz_max),
      m(other.m),
      n(other.n),
      row_start(other.row_start, other.row_start.stream()),
      j(other.j, other.j.stream()),
      x(other.x, other.x.stream())
  {
  }

  device_csr_matrix_t(const csr_matrix_t<i_t, f_t>& A, cuda::stream_ref stream)
    : m(A.m),
      n(A.n),
      nz_max(A.row_start[A.m]),
      row_start(A.row_start.size(), stream),
      j(A.j.size(), stream),
      x(A.x.size(), stream)
  {
    row_start = cuopt::device_copy(A.row_start, stream);
    j         = cuopt::device_copy(A.j, stream);
    x         = cuopt::device_copy(A.x, stream);
  }

  void resize_to_nnz(i_t nnz, cuda::stream_ref stream)
  {
    row_start.resize(m + 1, stream);
    j.resize(nnz, stream);
    x.resize(nnz, stream);
    nz_max = nnz;
  }

  csr_matrix_t<i_t, f_t> to_host(cuda::stream_ref stream)
  {
    csr_matrix_t<i_t, f_t> A(m, n, nz_max);
    A.row_start = cuopt::host_copy(row_start, stream);
    A.j         = cuopt::host_copy(j, stream);
    A.x         = cuopt::host_copy(x, stream);
    return A;
  }

  void copy(csr_matrix_t<i_t, f_t>& A, cuda::stream_ref stream)
  {
    m      = A.m;
    n      = A.n;
    nz_max = A.row_start[A.m];
    row_start.resize(A.row_start.size(), stream);
    raft::copy(row_start.data(), A.row_start.data(), A.row_start.size(), stream);
    j.resize(A.j.size(), stream);
    raft::copy(j.data(), A.j.data(), A.j.size(), stream);
    x.resize(A.x.size(), stream);
    raft::copy(x.data(), A.x.data(), A.x.size(), stream);
  }

  /** Copy from a device CSC matrix holding this matrix's transpose; the arrays are identical. */
  void copy_transposed(const device_csc_matrix_t<i_t, f_t>& AT, cuda::stream_ref stream)
  {
    m      = AT.n;
    n      = AT.m;
    nz_max = AT.nz_max;
    row_start.resize(AT.col_start.size(), stream);
    raft::copy(row_start.data(), AT.col_start.data(), AT.col_start.size(), stream);
    j.resize(AT.i.size(), stream);
    raft::copy(j.data(), AT.i.data(), AT.i.size(), stream);
    x.resize(AT.x.size(), stream);
    raft::copy(x.data(), AT.x.data(), AT.x.size(), stream);
  }

  i_t nz_max;                          // maximum number of entries
  i_t m;                               // number of rows
  i_t n;                               // number of columns
  rmm::device_uvector<i_t> row_start;  // row pointers (size m + 1)
  rmm::device_uvector<i_t> j;          // column indices, size nz_max
  rmm::device_uvector<f_t> x;          // numerical values, size nz_max

  static_assert(std::is_signed_v<i_t>);  // Require signed integers (we make use of this
                                         // to avoid extra space / computation)
};

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
  const i_t col_end = col_start[col + 1];
  for (i_t p = col_start[col] + static_cast<i_t>(threadIdx.x); p < col_end;
       p += static_cast<i_t>(blockDim.x)) {
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

template <typename i_t, typename f_t>
void device_csc_matrix_t<i_t, f_t>::to_compressed_row(device_csr_matrix_t<i_t, f_t>& Arow,
                                                      cuda::stream_ref stream) const
{
  Arow.m      = m;
  Arow.n      = n;
  Arow.nz_max = nz_max;
  Arow.row_start.resize(m + 1, stream);
  Arow.j.resize(nz_max, stream);
  Arow.x.resize(nz_max, stream);

  csc_to_csr_on_device<i_t, f_t>(m,
                                 n,
                                 nz_max,
                                 col_start.data(),
                                 i.data(),
                                 x.data(),
                                 Arow.row_start.data(),
                                 Arow.j.data(),
                                 Arow.x.data(),
                                 stream);
}

template <typename i_t, typename f_t>
void device_csc_matrix_t<i_t, f_t>::transpose(device_csc_matrix_t<i_t, f_t>& AT,
                                              cuda::stream_ref stream) const
{
  // A^T is n x m, and its CSC arrays are exactly the CSR arrays of A.
  AT.m      = n;
  AT.n      = m;
  AT.nz_max = nz_max;
  AT.col_start.resize(m + 1, stream);
  AT.i.resize(nz_max, stream);
  AT.x.resize(nz_max, stream);

  csc_to_csr_on_device<i_t, f_t>(m,
                                 n,
                                 nz_max,
                                 col_start.data(),
                                 i.data(),
                                 x.data(),
                                 AT.col_start.data(),
                                 AT.i.data(),
                                 AT.x.data(),
                                 stream);
}

}  // namespace cuopt::mathematical_optimization::barrier
