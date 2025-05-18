/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <thrust/binary_search.h>
#include <mip/problem/problem.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t, i_t MAX_EDGE_PER_CNST, typename view_t>
__device__ f_t spmv(view_t view, raft::device_span<f_t> input, i_t tid, i_t beg, i_t end)
{
  f_t out = 0.;
  for (i_t i = tid + beg; i < end; i += MAX_EDGE_PER_CNST) {
    auto coeff = view.coeff[i];
    auto var   = view.elem[i];
    auto in    = input[var];
    out += coeff * in;
  }
  return out;
}

template <typename i_t, typename f_t, i_t BDIM, typename view_t>
__global__ void lb_spmv_heavy_kernel(i_t id_range_beg,
                                     raft::device_span<const i_t> ids,
                                     raft::device_span<const i_t> pseudo_block_ids,
                                     i_t work_per_block,
                                     view_t view,
                                     raft::device_span<f_t> input,
                                     raft::device_span<f_t> tmp_out)
{
  auto idx             = ids[blockIdx.x] + id_range_beg;
  auto pseudo_block_id = pseudo_block_ids[blockIdx.x];
  i_t item_off_beg     = view.offsets[idx] + work_per_block * pseudo_block_id;
  i_t item_off_end     = min(item_off_beg + work_per_block, view.offsets[idx + 1]);

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto out = spmv<i_t, f_t, BDIM>(view, input, threadIdx.x, item_off_beg, item_off_end);

  out = BlockReduce(temp_storage).Sum(out);

  if (threadIdx.x == 0) { tmp_out[blockIdx.x] = out; }
}

template <typename i_t, typename f_t, typename view_t>
__global__ void finalize_spmv_kernel(i_t heavy_beg_id,
                                     raft::device_span<const i_t> item_offsets,
                                     raft::device_span<f_t> tmp_out,
                                     view_t view,
                                     raft::device_span<f_t> output)
{
  using warp_reduce = cub::WarpReduce<f_t>;
  __shared__ typename warp_reduce::TempStorage temp_storage;
  i_t idx      = heavy_beg_id + blockIdx.x;
  i_t item_idx = view.reorg_ids[idx];

  i_t item_off_beg = item_offsets[blockIdx.x];
  i_t item_off_end = item_offsets[blockIdx.x + 1];
  f_t out          = 0.;
  for (i_t i = threadIdx.x + item_off_beg; i < item_off_end; i += blockDim.x) {
    out += tmp_out[i];
  }
  out = warp_reduce(temp_storage).Sum(out);
  if (threadIdx.x == 0) { output[item_idx] = out; }
}

template <typename i_t, typename f_t, i_t BDIM, typename view_t>
__global__ void lb_spmv_block_kernel(i_t id_range_beg,
                                     view_t view,
                                     raft::device_span<f_t> input,
                                     raft::device_span<f_t> output)

{
  i_t idx          = id_range_beg + blockIdx.x;
  i_t item_idx     = view.reorg_ids[idx];
  i_t item_off_beg = view.offsets[idx];
  i_t item_off_end = view.offsets[idx + 1];

  typedef cub::BlockReduce<f_t, BDIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto out = spmv<i_t, f_t, BDIM>(view, input, threadIdx.x, item_off_beg, item_off_end);

  out = BlockReduce(temp_storage).Sum(out);

  if (threadIdx.x == 0) {
    // written to old index
    output[item_idx] = out;
  }
}

template <typename i_t, typename f_t, i_t BDIM, i_t MAX_EDGE_PER_CNST, typename view_t>
__global__ void lb_spmv_sub_warp_kernel(i_t id_range_beg,
                                        i_t id_range_end,
                                        view_t view,
                                        raft::device_span<f_t> input,
                                        raft::device_span<f_t> output)
{
  constexpr i_t ids_per_block = BDIM / MAX_EDGE_PER_CNST;
  i_t id_beg                  = blockIdx.x * ids_per_block + id_range_beg;
  i_t idx                     = id_beg + (threadIdx.x / MAX_EDGE_PER_CNST);
  i_t item_idx;
  if (idx < id_range_end) { item_idx = view.reorg_ids[idx]; }
  i_t p_tid = threadIdx.x % MAX_EDGE_PER_CNST;

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_CNST>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  f_t out = 0.;

  if (idx < id_range_end) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    out = spmv<i_t, f_t, MAX_EDGE_PER_CNST>(view, input, p_tid, item_off_beg, item_off_end);
  }

  out = warp_reduce(temp_storage).Sum(out);

  if (head_flag && (idx < id_range_end)) { output[item_idx] = out; }
}

#if 1

#define BYTE_TO_BINARY(byte)                                                               \
  ((byte) & 0x80 ? '1' : '0'), ((byte) & 0x40 ? '1' : '0'), ((byte) & 0x20 ? '1' : '0'),   \
    ((byte) & 0x10 ? '1' : '0'), ((byte) & 0x08 ? '1' : '0'), ((byte) & 0x04 ? '1' : '0'), \
    ((byte) & 0x02 ? '1' : '0'), ((byte) & 0x01 ? '1' : '0')

template <typename i_t>
__device__ __forceinline__ void get_sub_warp_bin(i_t* id_warp_beg,
                                                 i_t* id_range_end,
                                                 i_t* t_p_v,
                                                 raft::device_span<i_t> warp_offsets,
                                                 raft::device_span<i_t> bin_offsets)
{
  i_t warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
  i_t lane_id = threadIdx.x & 31;
  bool pred   = false;
  if (lane_id < warp_offsets.size()) { pred = (warp_id >= warp_offsets[lane_id]); }
  unsigned int m  = __ballot_sync(0xffffffff, pred);
  i_t seg         = 31 - __clz(m);
  i_t it_per_warp = (1 << (5 - seg));  // item per warp = 32/(2^seg)
  if (5 - seg < 0) {
    *t_p_v = 0;
    return;
  }
  i_t beg       = bin_offsets[seg] + (warp_id - warp_offsets[seg]) * it_per_warp;
  i_t end       = bin_offsets[seg + 1];
  *id_warp_beg  = beg;
  *id_range_end = end;
  *t_p_v        = (1 << seg);
}

template <typename i_t, typename f_t, i_t BDIM, i_t MAX_EDGE_PER_CNST, typename view_t>
__device__ void spmv_sub_warp(i_t id_warp_beg,
                              i_t id_range_end,
                              view_t view,
                              raft::device_span<f_t> input,
                              raft::device_span<f_t> output)
{
  i_t lane_id = (threadIdx.x & 31);
  i_t idx     = id_warp_beg + (lane_id / MAX_EDGE_PER_CNST);
  i_t item_idx;
  if (idx < id_range_end) { item_idx = view.reorg_ids[idx]; }
  i_t p_tid = lane_id & (MAX_EDGE_PER_CNST - 1);

  i_t head_flag = (p_tid == 0);

  using warp_reduce = cub::WarpReduce<f_t, MAX_EDGE_PER_CNST>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  f_t out = 0.;

  if (idx < id_range_end) {
    i_t item_off_beg = view.offsets[idx];
    i_t item_off_end = view.offsets[idx + 1];
    out = spmv<i_t, f_t, MAX_EDGE_PER_CNST>(view, input, p_tid, item_off_beg, item_off_end);
  }

  out = warp_reduce(temp_storage).Sum(out);

  if (head_flag && (idx < id_range_end)) { output[item_idx] = out; }
}

template <typename i_t, typename f_t, i_t BDIM, typename view_t>
__global__ void lb_spmv_sub_warp_kernel(view_t view,
                                        raft::device_span<f_t> input,
                                        raft::device_span<f_t> output,
                                        raft::device_span<i_t> warp_item_offsets,
                                        raft::device_span<i_t> warp_item_id_offsets)
{
  i_t id_warp_beg, id_range_end, t_p_v;
  get_sub_warp_bin<i_t>(
    &id_warp_beg, &id_range_end, &t_p_v, warp_item_offsets, warp_item_id_offsets);

  if (t_p_v == 1) {
    spmv_sub_warp<i_t, f_t, BDIM, 1>(id_warp_beg, id_range_end, view, input, output);
  } else if (t_p_v == 2) {
    spmv_sub_warp<i_t, f_t, BDIM, 2>(id_warp_beg, id_range_end, view, input, output);
  } else if (t_p_v == 4) {
    spmv_sub_warp<i_t, f_t, BDIM, 4>(id_warp_beg, id_range_end, view, input, output);
  } else if (t_p_v == 8) {
    spmv_sub_warp<i_t, f_t, BDIM, 8>(id_warp_beg, id_range_end, view, input, output);
  } else if (t_p_v == 16) {
    spmv_sub_warp<i_t, f_t, BDIM, 16>(id_warp_beg, id_range_end, view, input, output);
  }
}
#endif

}  // namespace cuopt::linear_programming::detail
