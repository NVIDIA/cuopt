/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <raft/core/device_span.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
inline __device__ bool check_infeasibility(f_t min_a, f_t max_a, f_t cnst_lb, f_t cnst_ub, f_t eps)
{
  return (min_a > cnst_ub + eps) || (max_a < cnst_lb - eps);
}

inline __device__ bool skip_both(thrust::pair<bool, bool>& skip_flag)
{
  return thrust::get<0>(skip_flag) && thrust::get<1>(skip_flag);
}

inline __device__ bool both_valid(thrust::pair<bool, bool>& skip_flag)
{
  return !thrust::get<0>(skip_flag) && !thrust::get<1>(skip_flag);
}

template <typename i_t>
__device__ __forceinline__ void get_sub_warp_bin(i_t* id_warp_beg,
                                                 i_t* id_range_end,
                                                 i_t* t_p_v,
                                                 raft::device_span<const i_t> warp_offsets,
                                                 raft::device_span<const i_t> warp_id_offsets,
                                                 i_t sub_warp_count)
{
  i_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
  i_t lane_id = threadIdx.x & 31;
  bool pred   = false;
  if (lane_id < warp_offsets.size()) { pred = (warp_id >= warp_offsets[lane_id]); }
  unsigned int m  = __ballot_sync(0xffffffff, pred);
  i_t seg         = 31 - __clz(m);
  i_t it_per_warp = (1 << (5 - seg));  // item per warp = 32/(2^seg)
  if ((5 - seg < 0) || (warp_id >= sub_warp_count)) {
    *t_p_v = 0;
    return;
  }
  i_t beg       = warp_id_offsets[seg] + (warp_id - warp_offsets[seg]) * it_per_warp;
  i_t end       = warp_id_offsets[seg + 1];
  *id_warp_beg  = beg;
  *id_range_end = end;
  *t_p_v        = (1 << seg);
}

template <typename i_t>
__device__ __forceinline__ void get_block_bin(i_t* id_block_beg,
                                              i_t* id_range_end,
                                              i_t* t_p_v,
                                              raft::device_span<const i_t> block_offsets,
                                              raft::device_span<const i_t> block_id_offsets,
                                              i_t sub_warp_block_count,
                                              i_t med_block_count)
{
  i_t lane_id       = threadIdx.x & 31;
  auto med_block_id = blockIdx.x - sub_warp_block_count;
  bool pred         = false;
  if (lane_id < block_offsets.size()) { pred = (med_block_id >= block_offsets[lane_id]); }
  unsigned int m      = __ballot_sync(0xffffffff, pred);
  i_t seg             = 31 - __clz(m);
  i_t threads_per_row = (32 << seg);
  // heavy
  if (threads_per_row > 256) {
    *t_p_v = threads_per_row;
    //*id_block_beg = sub_warp_block_count + med_block_count;
    //*id_range_end = gridDim.x;
    *id_block_beg = -1;
    *id_range_end = -1;
    return;
  } else {
    i_t beg =
      block_id_offsets[seg] + (med_block_id - block_offsets[seg]) * (blockDim.x / threads_per_row);
    i_t end = block_id_offsets[seg + 1];
    // if (threadIdx.x == 0) {
    //   printf("seg %d block_offsets[seg] %d block_id_offsets[seg] %d beg %d end %d\n", seg,
    //   block_offsets[seg], block_id_offsets[seg], beg, end);
    // }
    *id_block_beg = beg;
    *id_range_end = end;
    *t_p_v        = threads_per_row;
  }
}

template <typename f_t, int MAX_EDGE_PER_CNST, int BDIM>
struct warp_reduce_t {
  using f_t2        = typename type_2<f_t>::type;
  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_CNST>;

  // When LOGICAL_WARP_THREADS are a power of 2 then WarpReduce uses shuffle instead of shared
  // memory. Therefore, no shared memory is being used in these reductions
  using storage_t = typename warp_reduce::TempStorage[4 * BDIM / MAX_EDGE_PER_CNST];

  storage_t& temp_storage;

  __device__ warp_reduce_t(storage_t& storage_) : temp_storage(storage_)
  {
    static_assert(MAX_EDGE_PER_CNST && ((MAX_EDGE_PER_CNST & (MAX_EDGE_PER_CNST - 1)) == 0),
                  "MAX_EDGE_PER_CNST expected to be a power of 2");
  }

  inline __device__ thrust::pair<f_t2, f_t2> sum(thrust::pair<f_t2, f_t2> in)
  {
    f_t2 out0, out1;
    out0.x = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 0])
               .Sum(thrust::get<0>(in).x);
    out0.y = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 1])
               .Sum(thrust::get<0>(in).y);
    out1.x = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 2])
               .Sum(thrust::get<1>(in).x);
    out1.y = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 3])
               .Sum(thrust::get<1>(in).y);
    return thrust::make_pair(out0, out1);
  }

  inline __device__ thrust::pair<f_t2, f_t2> max_min(thrust::pair<f_t2, f_t2> in)
  {
    f_t2 out0, out1;
    out0.x = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 0])
               .Reduce(thrust::get<0>(in).x, cuda::maximum());
    out0.y = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 1])
               .Reduce(thrust::get<0>(in).y, cuda::minimum());
    out1.x = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 2])
               .Reduce(thrust::get<1>(in).x, cuda::maximum());
    out1.y = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 3])
               .Reduce(thrust::get<1>(in).y, cuda::minimum());
    return thrust::make_pair(out0, out1);
  }

  inline __device__ thrust::pair<f_t2, f_t2> sum(thrust::pair<f_t2, f_t2> in, int valid_items)
  {
    f_t2 out0, out1;
    out0.x = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 0])
               .Sum(thrust::get<0>(in).x, valid_items);
    out0.y = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 1])
               .Sum(thrust::get<0>(in).y, valid_items);
    out1.x = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 2])
               .Sum(thrust::get<1>(in).x, valid_items);
    out1.y = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 3])
               .Sum(thrust::get<1>(in).y, valid_items);
    return thrust::make_pair(out0, out1);
  }

  inline __device__ thrust::pair<f_t2, f_t2> max_min(thrust::pair<f_t2, f_t2> in, int valid_items)
  {
    f_t2 out0, out1;
    out0.x = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 0])
               .Reduce(thrust::get<0>(in).x, cuda::maximum(), valid_items);
    out0.y = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 1])
               .Reduce(thrust::get<0>(in).y, cuda::minimum(), valid_items);
    out1.x = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 2])
               .Reduce(thrust::get<1>(in).x, cuda::maximum(), valid_items);
    out1.y = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 3])
               .Reduce(thrust::get<1>(in).y, cuda::minimum(), valid_items);
    return thrust::make_pair(out0, out1);
  }

  inline __device__ f_t2 max_min(f_t2& in)
  {
    f_t2 out;
    out.x = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 0])
              .Reduce(in.x, cuda::maximum());
    out.y = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 1])
              .Reduce(in.y, cuda::minimum());
    return out;
  }

  inline __device__ f_t2 max_min(f_t2& in, int valid_items)
  {
    f_t2 out;
    out.x = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 0])
              .Reduce(in.x, cuda::maximum(), valid_items);
    out.y = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 1])
              .Reduce(in.y, cuda::minimum(), valid_items);
    return out;
  }

  inline __device__ f_t2 sum(f_t2& in)
  {
    f_t2 out;
    out.x = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 0]).Sum(in.x);
    out.y = warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 1]).Sum(in.y);
    return out;
  }

  inline __device__ f_t2 sum(f_t2& in, int valid_items)
  {
    f_t2 out;
    out.x =
      warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 0]).Sum(in.x, valid_items);
    out.y =
      warp_reduce(temp_storage[4 * (threadIdx.x / MAX_EDGE_PER_CNST) + 1]).Sum(in.y, valid_items);
    return out;
  }
};

template <typename f_t, int BDIM>
struct block_reduce_t {
  using f_t2         = typename type_2<f_t>::type;
  using block_reduce = cub::BlockReduce<f_t, BDIM>;
  using storage_t    = typename block_reduce::TempStorage;

  storage_t& temp_storage;

  __device__ block_reduce_t(storage_t& storage_) : temp_storage(storage_) {}

  inline __device__ f_t2 max_min(f_t2& in)
  {
    f_t2 out;
    out.x = block_reduce(temp_storage).Reduce(in.x, cuda::maximum());
    __syncthreads();
    out.y = block_reduce(temp_storage).Reduce(in.y, cuda::minimum());
    return out;
  }

  inline __device__ thrust::pair<f_t2, f_t2> max_min(thrust::pair<f_t2, f_t2>& in)
  {
    f_t2 out0, out1;
    out0.x = block_reduce(temp_storage).Reduce(thrust::get<0>(in).x, cuda::maximum());
    __syncthreads();
    out0.y = block_reduce(temp_storage).Reduce(thrust::get<0>(in).y, cuda::minimum());
    __syncthreads();
    out1.x = block_reduce(temp_storage).Reduce(thrust::get<1>(in).x, cuda::maximum());
    __syncthreads();
    out1.y = block_reduce(temp_storage).Reduce(thrust::get<1>(in).y, cuda::minimum());
    return thrust::make_pair(out0, out1);
  }

  inline __device__ f_t2 sum(f_t2& in)
  {
    f_t2 out;
    out.x = block_reduce(temp_storage).Sum(in.x);
    __syncthreads();
    out.y = block_reduce(temp_storage).Sum(in.y);
    return out;
  }

  inline __device__ thrust::pair<f_t2, f_t2> sum(thrust::pair<f_t2, f_t2>& in)
  {
    f_t2 out0, out1;
    out0.x = block_reduce(temp_storage).Sum(thrust::get<0>(in).x);
    __syncthreads();
    out0.y = block_reduce(temp_storage).Sum(thrust::get<0>(in).y);
    __syncthreads();
    out1.x = block_reduce(temp_storage).Sum(thrust::get<1>(in).x);
    __syncthreads();
    out1.y = block_reduce(temp_storage).Sum(thrust::get<1>(in).y);
    return thrust::make_pair(out0, out1);
  }
};

template <typename f_t, int BDIM, int PSEUDO_BDIM>
struct partial_block_reduce_t {
  using f_t2 = typename type_2<f_t>::type;

  using reduce_t = warp_reduce_t<f_t, 32, BDIM>;

  struct storage_t {
    using warp_reduce_storage_t = typename reduce_t::storage_t;
    warp_reduce_storage_t warp_storage[BDIM / raft::WarpSize];
    f_t2 act0[BDIM / raft::WarpSize];
    f_t2 act1[BDIM / raft::WarpSize];
  };

  storage_t& temp_storage;

  __device__ partial_block_reduce_t(storage_t& storage_) : temp_storage(storage_) {};

  inline __device__ bool is_aggregated_thread() { return (threadIdx.x & (PSEUDO_BDIM - 1)) == 0; }

  inline __device__ int pseudo_thread_id() { return (threadIdx.x & (PSEUDO_BDIM - 1)); }

  inline __device__ thrust::pair<f_t2, f_t2> max_min(thrust::pair<f_t2, f_t2>& in)
  {
    int warp_id = threadIdx.x / raft::WarpSize;

    reduce_t reduce(temp_storage.warp_storage[warp_id]);
    auto warp_agg = reduce.max_min(in);

    // write temps to shared memory
    if ((threadIdx.x & (raft::WarpSize - 1)) == 0) {
      temp_storage.act0[warp_id] = thrust::get<0>(warp_agg);
      temp_storage.act1[warp_id] = thrust::get<1>(warp_agg);
    }
    __syncthreads();

    auto bounds = thrust::make_pair(
      f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()},
      f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()});
    // in 'head warp' of each pseudo block, load temp results of children warps
    // and sum them up
    if ((threadIdx.x & (PSEUDO_BDIM - 1)) / raft::WarpSize == 0) {
      constexpr int valid_item_count = PSEUDO_BDIM / raft::WarpSize;
      static_assert(valid_item_count < raft::WarpSize,
                    "number of valid items cannot exceed warpsize");
      auto lane_id = threadIdx.x & (raft::WarpSize - 1);
      if (lane_id < valid_item_count) {
        auto temp_warp_result_idx = lane_id + warp_id;
        bounds                    = thrust::make_pair(temp_storage.act0[temp_warp_result_idx],
                                   temp_storage.act1[temp_warp_result_idx]);
      }
      bounds = reduce.max_min(bounds, valid_item_count);
    }
    return bounds;
  }

  inline __device__ thrust::pair<f_t2, f_t2> sum(thrust::pair<f_t2, f_t2>& in)
  {
    int warp_id = threadIdx.x / raft::WarpSize;

    reduce_t reduce(temp_storage.warp_storage[warp_id]);
    auto warp_sum = reduce.sum(in);

    // write temps to shared memory
    if ((threadIdx.x & (raft::WarpSize - 1)) == 0) {
      temp_storage.act0[warp_id] = thrust::get<0>(warp_sum);
      temp_storage.act1[warp_id] = thrust::get<1>(warp_sum);
    }
    __syncthreads();

    auto act = thrust::make_pair(f_t2{0., 0.}, f_t2{0., 0.});
    // in 'head warp' of each pseudo block, load temp results of children warps
    // and sum them up
    if ((threadIdx.x & (PSEUDO_BDIM - 1)) / raft::WarpSize == 0) {
      constexpr int valid_item_count = PSEUDO_BDIM / raft::WarpSize;
      static_assert(valid_item_count < raft::WarpSize,
                    "number of valid items cannot exceed warpsize");
      auto lane_id = threadIdx.x & (raft::WarpSize - 1);
      if (lane_id < valid_item_count) {
        auto temp_warp_result_idx = lane_id + warp_id;
        act                       = thrust::make_pair(temp_storage.act0[temp_warp_result_idx],
                                temp_storage.act1[temp_warp_result_idx]);
      }
      act = reduce.sum(act, valid_item_count);
    }
    return act;
  }

  inline __device__ f_t2 sum(f_t2& in)
  {
    int warp_id = threadIdx.x / raft::WarpSize;

    reduce_t reduce(temp_storage.warp_storage[warp_id]);
    auto warp_sum = reduce.sum(in);

    // write temps to shared memory
    if ((threadIdx.x & (raft::WarpSize - 1)) == 0) { temp_storage.act0[warp_id] = warp_sum; }
    __syncthreads();

    auto act = f_t2{0., 0.};
    // in 'head warp' of each pseudo block, load temp results of children warps
    // and sum them up
    if ((threadIdx.x & (PSEUDO_BDIM - 1)) / raft::WarpSize == 0) {
      constexpr int valid_item_count = PSEUDO_BDIM / raft::WarpSize;
      static_assert(valid_item_count < raft::WarpSize,
                    "number of valid items cannot exceed warpsize");
      auto lane_id = threadIdx.x & (raft::WarpSize - 1);
      if (lane_id < valid_item_count) {
        auto temp_warp_result_idx = lane_id + warp_id;
        act                       = temp_storage.act0[temp_warp_result_idx];
      }
      act = reduce.sum(act, valid_item_count);
    }
    return act;
  }

  inline __device__ f_t2 max_min(f_t2& in)
  {
    int warp_id = threadIdx.x / raft::WarpSize;

    reduce_t reduce(temp_storage.warp_storage[warp_id]);
    auto warp_agg = reduce.max_min(in);

    // write temps to shared memory
    if ((threadIdx.x & (raft::WarpSize - 1)) == 0) { temp_storage.act0[warp_id] = warp_agg; }
    __syncthreads();

    auto bounds = f_t2{-std::numeric_limits<f_t>::infinity(), std::numeric_limits<f_t>::infinity()};
    // in 'head warp' of each pseudo block, load temp results of children warps
    // and sum them up
    if ((threadIdx.x & (PSEUDO_BDIM - 1)) / raft::WarpSize == 0) {
      constexpr int valid_item_count = PSEUDO_BDIM / raft::WarpSize;
      static_assert(valid_item_count < raft::WarpSize,
                    "number of valid items cannot exceed warpsize");
      auto lane_id = threadIdx.x & (raft::WarpSize - 1);
      if (lane_id < valid_item_count) {
        auto temp_warp_result_idx = lane_id + warp_id;
        bounds                    = temp_storage.act0[temp_warp_result_idx];
      }
      bounds = reduce.max_min(bounds, valid_item_count);
    }
    return bounds;
  }
};

template <int BDIM>
struct vote_storage_t {
  bool changed_0[BDIM / raft::WarpSize];
  bool changed_1[BDIM / raft::WarpSize];
};

template <typename f_t, int BDIM>
union reduction_storage_t {
  typename block_reduce_t<f_t, BDIM>::storage_t block_reduce_storage_;

  typename warp_reduce_t<f_t, 32, BDIM>::storage_t warp_storage_32_;
  typename warp_reduce_t<f_t, 16, BDIM>::storage_t warp_storage_16_;
  typename warp_reduce_t<f_t, 8, BDIM>::storage_t warp_storage_8_;
  typename warp_reduce_t<f_t, 4, BDIM>::storage_t warp_storage_4_;
  typename warp_reduce_t<f_t, 2, BDIM>::storage_t warp_storage_2_;
  typename warp_reduce_t<f_t, 1, BDIM>::storage_t warp_storage_1_;

  typename partial_block_reduce_t<f_t, BDIM, 256>::storage_t partial_block_storage_256_;
  typename partial_block_reduce_t<f_t, BDIM, 128>::storage_t partial_block_storage_128_;
  typename partial_block_reduce_t<f_t, BDIM, 64>::storage_t partial_block_storage_64_;
  vote_storage_t<BDIM> vote;
};

template <int size, typename storage_t>
__device__ auto& warp_storage(storage_t& storage)
{
  if constexpr (size == 32) {
    return storage.warp_storage_32_;
  } else if constexpr (size == 16) {
    return storage.warp_storage_16_;
  } else if constexpr (size == 8) {
    return storage.warp_storage_8_;
  } else if constexpr (size == 4) {
    return storage.warp_storage_4_;
  } else if constexpr (size == 2) {
    return storage.warp_storage_2_;
  } else if constexpr (size == 1) {
    return storage.warp_storage_1_;
  }
}

template <int size, typename storage_t>
__device__ auto& partial_block_storage(storage_t& storage)
{
  if constexpr (size == 256) {
    return storage.partial_block_storage_256_;
  } else if constexpr (size == 128) {
    return storage.partial_block_storage_128_;
  } else if constexpr (size == 64) {
    return storage.partial_block_storage_64_;
  }
}

template <typename storage_t>
__device__ auto& block_storage(storage_t& storage)
{
  return storage.block_reduce_storage_;
}

}  // namespace cuopt::linear_programming::detail
