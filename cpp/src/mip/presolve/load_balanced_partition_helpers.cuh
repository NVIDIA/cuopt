/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <utilities/copy_helpers.hpp>

namespace cuopt::linear_programming::detail {

template <typename T>
struct type_2 {
  using type = void;
};

template <>
struct type_2<int> {
  using type = int2;
};

template <>
struct type_2<float> {
  using type = float2;
};

template <>
struct type_2<double> {
  using type = double2;
};

template <typename T>
raft::device_span<typename type_2<T>::type> make_span_2(rmm::device_uvector<T>& container)
{
  // TODO : ceildiv or throw assert
  using T2 = typename type_2<T>::type;
  return raft::device_span<T2>(reinterpret_cast<T2*>(container.data()),
                               sizeof(T) * container.size() / sizeof(T2));
}

template <typename T>
raft::device_span<const typename type_2<T>::type> make_span_2(
  rmm::device_uvector<T> const& container)
{
  // TODO : ceildiv or throw assert
  using T2 = typename type_2<T>::type;
  return raft::device_span<const T2>(reinterpret_cast<const T2*>(container.data()),
                                     sizeof(T) * container.size() / sizeof(T2));
}

template <typename degree_t>
constexpr int BitsPWrd = sizeof(degree_t) * 8;

template <typename degree_t>
constexpr int NumberBins = sizeof(degree_t) * 8 + 1;

template <typename i_t>
int ceil_log_2(i_t val)
{
  return BitsPWrd<i_t> - __builtin_clz(val) + (__builtin_popcount(val) > 1);
}

template <typename degree_t>
__device__ inline typename std::enable_if<(sizeof(degree_t) == 4), int>::type ceilLog2_p1(
  degree_t val)
{
  auto bin_number = BitsPWrd<degree_t> - __clz(val) + (__popc(val) > 1);
  return bin_number;
}

template <typename degree_t>
__device__ inline typename std::enable_if<(sizeof(degree_t) == 8), int>::type ceilLog2_p1(
  degree_t val)
{
  auto bin_number = BitsPWrd<degree_t> - __clzll(val) + (__popcll(val) > 1);
  return bin_number;
}

template <typename T>
__global__ void exclusive_scan(T* data, T* out)
{
  constexpr int BinCount = NumberBins<T>;
  T lData[BinCount];
  thrust::exclusive_scan(thrust::seq, data, data + BinCount, lData);
  for (int i = 0; i < BinCount; ++i) {
    out[i]  = lData[i];
    data[i] = lData[i];
  }
}

// Return true if the nth bit of an array is set to 1
template <typename T>
__device__ bool is_nth_bit_set(unsigned* bitmap, T index)
{
  return bitmap[index / BitsPWrd<unsigned>] & (unsigned{1} << (index % BitsPWrd<unsigned>));
}

// Given the CSR offsets of vertices and the related active bit map
// count the number of vertices that belong to a particular bin where
// vertex with degree d such that 2^x < d <= 2^x+1 belong to bin (x+1)
// Vertices with degree 0 are counted in bin 0
template <typename i_t>
__global__ void count_bin_sizes(
  i_t* bins, i_t const* offsets, i_t vertex_begin, i_t vertex_end, unsigned* active_bitmap)
{
  constexpr int BinCount = NumberBins<i_t>;
  __shared__ i_t lBin[BinCount];
  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) {
    lBin[i] = 0;
  }
  __syncthreads();

  if (active_bitmap == nullptr) {
    for (i_t i = threadIdx.x + (blockIdx.x * blockDim.x); i < (vertex_end - vertex_begin);
         i += gridDim.x * blockDim.x) {
      atomicAdd(lBin + ceilLog2_p1(offsets[i + 1] - offsets[i]), i_t{1});
    }
  } else {
    for (i_t i = threadIdx.x + (blockIdx.x * blockDim.x); i < (vertex_end - vertex_begin);
         i += gridDim.x * blockDim.x) {
      if (is_nth_bit_set(active_bitmap, vertex_begin + i)) {
        atomicAdd(lBin + ceilLog2_p1(offsets[i + 1] - offsets[i]), i_t{1});
      }
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < BinCount; i += blockDim.x) {
    atomicAdd(bins + i, lBin[i]);
  }
}

// Bin vertices to the appropriate bins by taking into account
// the starting offsets calculated by count_bin_sizes
template <typename i_t>
__global__ void create_vertex_bins(i_t* reorg_vertices,
                                   i_t* bin_offsets,
                                   i_t const* offsets,
                                   i_t vertex_begin,
                                   i_t vertex_end,
                                   unsigned* active_bitmap)
{
  constexpr int BinCount = NumberBins<i_t>;
  __shared__ i_t lBin[BinCount];
  __shared__ int lPos[BinCount];
  if (threadIdx.x < BinCount) {
    lBin[threadIdx.x] = 0;
    lPos[threadIdx.x] = 0;
  }
  __syncthreads();

  i_t vertex_id        = (threadIdx.x + blockIdx.x * blockDim.x);
  bool is_valid_vertex = (vertex_id < (vertex_end - vertex_begin));
  if (active_bitmap != nullptr) {
    is_valid_vertex = is_valid_vertex && (is_nth_bit_set(active_bitmap, vertex_begin + vertex_id));
  }

  int threadBin;
  i_t threadPos;
  if (is_valid_vertex) {
    threadBin = ceilLog2_p1(offsets[vertex_id + 1] - offsets[vertex_id]);
    threadPos = atomicAdd(lBin + threadBin, i_t{1});
  }
  __syncthreads();

  if (threadIdx.x < BinCount) {
    lPos[threadIdx.x] = atomicAdd(bin_offsets + threadIdx.x, lBin[threadIdx.x]);
  }
  __syncthreads();

  if (is_valid_vertex) { reorg_vertices[lPos[threadBin] + threadPos] = vertex_id; }
}

template <typename i_t>
void bin_vertices(rmm::device_uvector<i_t>& reorg_vertices,
                  rmm::device_uvector<i_t>& bin_count_offsets,
                  rmm::device_uvector<i_t>& bin_count,
                  unsigned* active_bitmap,
                  const i_t* offsets,
                  i_t vertex_begin,
                  i_t vertex_end,
                  cudaStream_t stream)
{
  const unsigned BLOCK_SIZE = 512;
  unsigned blocks           = ((vertex_end - vertex_begin) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  count_bin_sizes<i_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
    bin_count.data(), offsets, vertex_begin, vertex_end, active_bitmap);

  exclusive_scan<<<1, 1, 0, stream>>>(bin_count.data(), bin_count_offsets.data());

  i_t vertex_count = bin_count.back_element(stream);
  reorg_vertices.resize(vertex_count, stream);

  create_vertex_bins<i_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
    reorg_vertices.data(), bin_count.data(), offsets, vertex_begin, vertex_end, active_bitmap);
}

template <typename i_t>
struct id_bucket_t {
  raft::device_span<i_t> vertex_ids;
  i_t ceilLogDegreeStart;
  i_t ceilLogDegreeEnd;
};

template <typename i_t>
class log_dist_t {
  i_t* vertex_id_begin_;

 public:
  std::vector<i_t> bin_offsets_;

  log_dist_t() = default;

  log_dist_t(rmm::device_uvector<i_t>& vertex_id, rmm::device_uvector<i_t>& bin_offsets)
    : vertex_id_begin_(vertex_id.data()), bin_offsets_(host_copy(bin_offsets))
  {
    // If bin_offsets_ is smaller than NumberBins<i_t> then resize it
    // so that the last element is repeated
    bin_offsets_.resize(NumberBins<i_t>, bin_offsets_.back());
  }

  id_bucket_t<i_t> degree_range(i_t ceilLogDegreeStart = -1,
                                i_t ceilLogDegreeEnd   = std::numeric_limits<i_t>::max())
  {
    if (ceilLogDegreeEnd > static_cast<i_t>(bin_offsets_.size()) - 2) {
      ceilLogDegreeEnd = bin_offsets_.size() - 2;
    }
    return id_bucket_t<i_t>{
      raft::device_span<i_t>{vertex_id_begin_ + bin_offsets_[ceilLogDegreeStart + 1],
                             static_cast<size_t>(bin_offsets_[ceilLogDegreeEnd + 1] -
                                                 bin_offsets_[ceilLogDegreeStart + 1])},
      ceilLogDegreeStart,
      ceilLogDegreeEnd};
  }

  std::pair<raft::device_span<i_t>, raft::device_span<i_t>> partition_ids(i_t beg_ceil_log_2)
  {
    beg_ceil_log_2     = std::max(beg_ceil_log_2, i_t{0});
    i_t end_ceil_log_2 = bin_offsets_.size() - 2;

    auto light = raft::device_span<i_t>{
      vertex_id_begin_ + bin_offsets_[5 + 1],
      static_cast<size_t>(bin_offsets_[beg_ceil_log_2 + 1] - bin_offsets_[5 + 1])};
    auto heavy = raft::device_span<i_t>{
      vertex_id_begin_ + bin_offsets_[beg_ceil_log_2 + 1],
      static_cast<size_t>(bin_offsets_[end_ceil_log_2 + 1] - bin_offsets_[beg_ceil_log_2 + 1])};
    return std::make_pair(light, heavy);
  }

  raft::device_span<i_t> get_heavy_ids(i_t beg_ceil_log_2)
  {
    beg_ceil_log_2     = std::max(beg_ceil_log_2, i_t{0});
    i_t end_ceil_log_2 = bin_offsets_.size() - 2;

    return raft::device_span<i_t>{
      vertex_id_begin_ + bin_offsets_[beg_ceil_log_2 + 1],
      bin_offsets_[end_ceil_log_2 + 1] - bin_offsets_[beg_ceil_log_2 + 1]};
  }
};

template <typename i_t>
class vertex_bin_t {
  const i_t* offsets_;
  unsigned* active_bitmap_;
  i_t vertex_begin_;
  i_t vertex_end_;

  rmm::device_uvector<i_t> tempBins_;
  rmm::device_uvector<i_t> bin_offsets_;

 public:
  vertex_bin_t(const raft::handle_t* handle_ptr)
    : tempBins_(NumberBins<i_t>, handle_ptr->get_stream()),
      bin_offsets_(NumberBins<i_t>, handle_ptr->get_stream())
  {
    thrust::fill(handle_ptr->get_thrust_policy(), bin_offsets_.begin(), bin_offsets_.end(), 0);
    thrust::fill(handle_ptr->get_thrust_policy(), tempBins_.begin(), tempBins_.end(), 0);
  }

  void setup(const i_t* offsets, unsigned* active_bitmap, i_t vertex_begin, i_t vertex_end)
  {
    offsets_       = offsets;
    active_bitmap_ = active_bitmap;
    vertex_begin_  = vertex_begin;
    vertex_end_    = vertex_end;
  }

  log_dist_t<i_t> run(rmm::device_uvector<i_t>& reorganized_vertices,
                      const raft::handle_t* handle_ptr);
};

template <typename i_t>
log_dist_t<i_t> vertex_bin_t<i_t>::run(rmm::device_uvector<i_t>& reorganized_vertices,
                                       const raft::handle_t* handle_ptr)
{
  thrust::fill(handle_ptr->get_thrust_policy(), bin_offsets_.begin(), bin_offsets_.end(), i_t{0});
  thrust::fill(handle_ptr->get_thrust_policy(), tempBins_.begin(), tempBins_.end(), i_t{0});
  bin_vertices(reorganized_vertices,
               bin_offsets_,
               tempBins_,
               active_bitmap_,
               offsets_,
               vertex_begin_,
               vertex_end_,
               handle_ptr->get_stream());

  return log_dist_t<i_t>(reorganized_vertices, bin_offsets_);
}

}  // namespace cuopt::linear_programming::detail
