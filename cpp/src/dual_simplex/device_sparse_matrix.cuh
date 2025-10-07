/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>

#include <cub/cub.cuh>
#include <rmm/device_scalar.hpp>
#include <rmm/device_vector.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/cuda_helpers.cuh>

namespace cuopt::linear_programming::dual_simplex {

template <typename f_t>
struct sum_reduce_helper_t {
  rmm::device_buffer buffer_data;
  rmm::device_scalar<f_t> out;
  size_t buffer_size;

  sum_reduce_helper_t(rmm::cuda_stream_view stream_view)
    : buffer_data(0, stream_view), out(stream_view)
  {
  }

  template <typename InputIteratorT, typename i_t>
  f_t sum(InputIteratorT input, i_t size, rmm::cuda_stream_view stream_view)
  {
    buffer_size = 0;
    cub::DeviceReduce::Sum(nullptr, buffer_size, input, out.data(), size, stream_view);
    buffer_data.resize(buffer_size, stream_view);
    cub::DeviceReduce::Sum(buffer_data.data(), buffer_size, input, out.data(), size, stream_view);
    return out.value(stream_view);
  }
};

template <typename f_t>
struct transform_reduce_helper_t {
  rmm::device_buffer buffer_data;
  rmm::device_scalar<f_t> out;
  size_t buffer_size;

  transform_reduce_helper_t(rmm::cuda_stream_view stream_view)
    : buffer_data(0, stream_view), out(stream_view)
  {
  }

  template <typename InputIteratorT, typename ReductionOpT, typename TransformOpT, typename i_t>
  f_t transform_reduce(InputIteratorT input,
                       ReductionOpT reduce_op,
                       TransformOpT transform_op,
                       f_t init,
                       i_t size,
                       rmm::cuda_stream_view stream_view)
  {
    cub::DeviceReduce::TransformReduce(
      nullptr, buffer_size, input, out.data(), size, reduce_op, transform_op, init, stream_view);

    buffer_data.resize(buffer_size, stream_view);

    cub::DeviceReduce::TransformReduce(buffer_data.data(),
                                       buffer_size,
                                       input,
                                       out.data(),
                                       size,
                                       reduce_op,
                                       transform_op,
                                       init,
                                       stream_view);

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
  device_csc_matrix_t(rmm::cuda_stream_view stream)
    : col_start(0, stream), i(0, stream), x(0, stream), col_index(0, stream)
  {
  }

  device_csc_matrix_t(i_t rows, i_t cols, i_t nz, rmm::cuda_stream_view stream)
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

  device_csc_matrix_t(const csc_matrix_t<i_t, f_t>& A, rmm::cuda_stream_view stream)
    : m(A.m),
      n(A.n),
      nz_max(A.col_start[A.n]),
      col_start(A.col_start.size(), stream),
      i(A.i.size(), stream),
      x(A.x.size(), stream)
  {
    col_start = cuopt::device_copy(A.col_start, stream);
    i         = cuopt::device_copy(A.i, stream);
    x         = cuopt::device_copy(A.x, stream);
  }

  void resize_to_nnz(i_t nnz, rmm::cuda_stream_view stream)
  {
    col_start.resize(n + 1, stream);
    i.resize(nnz, stream);
    x.resize(nnz, stream);
    nz_max = nnz;
  }

  csc_matrix_t<i_t, f_t> to_host(rmm::cuda_stream_view stream)
  {
    csc_matrix_t<i_t, f_t> A(m, n, nz_max);
    A.col_start = cuopt::host_copy(col_start, stream);
    A.i         = cuopt::host_copy(i, stream);
    A.x         = cuopt::host_copy(x, stream);
    return A;
  }

  void copy(csc_matrix_t<i_t, f_t>& A, rmm::cuda_stream_view stream)
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

  void form_col_index(rmm::cuda_stream_view stream)
  {
    col_index.resize(x.size(), stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(col_index.data(), 0, sizeof(i_t) * col_index.size(), stream));
    // Scatter 1 when there is a col start in col_index
    if (col_start.size() > 2) {
      thrust::for_each(rmm::exec_policy(stream),
                       thrust::make_counting_iterator(i_t(1)),  // Skip the first 0
                       thrust::make_counting_iterator(
                         static_cast<i_t>(col_start.size() - 1)),  // Skip the end index
                       [span_col_start = cuopt::make_span(col_start),
                        span_col_index = cuopt::make_span(col_index)] __device__(i_t i) {
                         span_col_index[span_col_start[i]] = 1;
                       });
    }

    // Inclusive cumulative sum to have the corresponding column for each entry
    rmm::device_buffer d_temp_storage;
    size_t temp_storage_bytes{0};
    cub::DeviceScan::InclusiveSum(
      nullptr, temp_storage_bytes, col_index.data(), col_index.data(), col_index.size(), stream);
    d_temp_storage.resize(temp_storage_bytes, stream);
    cub::DeviceScan::InclusiveSum(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  col_index.data(),
                                  col_index.data(),
                                  col_index.size(),
                                  stream);
    // Have to sync since InclusiveSum is being run on local data (d_temp_storage)
    stream.synchronize();
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
  device_csr_matrix_t(rmm::cuda_stream_view stream)
    : row_start(0, stream), j(0, stream), x(0, stream)
  {
  }

  device_csr_matrix_t(i_t rows, i_t cols, i_t nz, rmm::cuda_stream_view stream)
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

  device_csr_matrix_t(const csr_matrix_t<i_t, f_t>& A, rmm::cuda_stream_view stream)
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

  void resize_to_nnz(i_t nnz, rmm::cuda_stream_view stream)
  {
    row_start.resize(m + 1, stream);
    j.resize(nnz, stream);
    x.resize(nnz, stream);
    nz_max = nnz;
  }

  csr_matrix_t<i_t, f_t> to_host(rmm::cuda_stream_view stream)
  {
    csr_matrix_t<i_t, f_t> A(m, n, nz_max);
    A.row_start = cuopt::host_copy(row_start, stream);
    A.j         = cuopt::host_copy(j, stream);
    A.x         = cuopt::host_copy(x, stream);
    return A;
  }

  void copy(csr_matrix_t<i_t, f_t>& A, rmm::cuda_stream_view stream)
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

  i_t nz_max;                          // maximum number of entries
  i_t m;                               // number of rows
  i_t n;                               // number of columns
  rmm::device_uvector<i_t> row_start;  // row pointers (size m + 1)
  rmm::device_uvector<i_t> j;          // column indices, size nz_max
  rmm::device_uvector<f_t> x;          // numerical values, size nz_max

  static_assert(std::is_signed_v<i_t>);  // Require signed integers (we make use of this
                                         // to avoid extra space / computation)
};

}  // namespace cuopt::linear_programming::dual_simplex
